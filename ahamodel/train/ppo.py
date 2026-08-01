"""PPO 从零实现：actor + 冻结 reference + RM + critic(GAE)，单卡可跑。"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from ahamodel.config import ModelConfig, StageConfig
from ahamodel.data.dataset import collate_rollout
from ahamodel.model.generate import generate_batch
from ahamodel.model.model import AhaForCausalLM
from ahamodel.train.rl import compute_gae, make_response_labels, per_token_kl, response_logprobs
from ahamodel.train.reward_model import RewardModel


class CriticModel(nn.Module):
    """PPO 值函数：骨干 + 逐 token 标量头（GAE 需要每个 token 的 value）。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AhaForCausalLM(cfg)
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.model(
            input_ids,
            use_cache=False,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return self.value_head(out.hidden_states).squeeze(-1)  # (b, s)


def _pad_to(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


class PPOTrainer:
    def __init__(
        self,
        actor: AhaForCausalLM,
        critic: CriticModel,
        rm: RewardModel,
        reference: AhaForCausalLM,
        cfg: StageConfig,
        device: str,
        logger=None,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.rm = rm.to(device)
        self.reference = reference.to(device).eval()
        self.rm.eval()
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.step = 0
        self.actor_opt = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            [p for p in critic.parameters() if p.requires_grad], lr=cfg.lr * 2, weight_decay=cfg.weight_decay
        )

    def rollout(self, prompts: List[List[int]], tokenizer) -> List[List[int]]:
        """采样响应（返回每条的新增 token 序列）。"""
        return generate_batch(
            self.actor,
            tokenizer,
            prompts,
            max_new_tokens=self.cfg.response_max_len,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            device=self.device,
        )

    def step_once(self, prompts: List[List[int]], responses: List[List[int]], tokenizer) -> dict:
        """对一批 (prompt, response) 做一次 PPO 更新。"""
        cfg = self.cfg
        pad_id = tokenizer.pad_id or 0
        device = self.device

        full_ids = [torch.tensor(p + r, dtype=torch.long) for p, r in zip(prompts, responses)]
        starts = [len(p) for p in prompts]
        resp_lens = [len(r) for r in responses]
        if any(n == 0 for n in resp_lens):
            return {"loss": 0.0, "reward": 0.0}
        T = max(resp_lens)
        ids_t = _pad_to(full_ids, pad_id).to(device)
        b, s = ids_t.shape
        mask_t = torch.zeros_like(ids_t)
        for i, ids in enumerate(full_ids):
            mask_t[i, : len(ids)] = 1
        starts_t = torch.tensor(starts, device=device)
        lens_t = torch.tensor(resp_lens, device=device)
        labels_t = make_response_labels(ids_t, starts_t, lens_t, pad_id)

        with torch.no_grad():
            ref_sum, ref_token = response_logprobs(self.reference, ids_t, labels_t, device)
            old_sum, old_token = response_logprobs(self.actor, ids_t, labels_t, device)
            rm_rewards = self.rm(ids_t, starts_t, mask_t)  # (b,)
            values_full = self.critic(ids_t, mask_t)

        # 逐 token reward：每步 -kl_coef*KL，最后一个响应 token 额外加 RM 奖励
        kl_token = per_token_kl(ref_token, old_token, mode="k2")
        rewards = torch.zeros(b, T, device=device)
        values = torch.zeros(b, T, device=device)
        masks = torch.zeros(b, T, dtype=torch.long, device=device)
        kl_vals = []
        for i in range(b):
            st, ln = starts[i], resp_lens[i]
            pos = slice(st - 1, st - 1 + ln)
            rewards[i, :ln] = -cfg.kl_coef * kl_token[i, pos]
            rewards[i, ln - 1] += rm_rewards[i]
            values[i, :ln] = values_full[i, pos]
            masks[i, :ln] = 1
            kl_vals.append(kl_token[i, pos])
        kl_mean = torch.cat(kl_vals).mean()

        advantages, returns = compute_gae(rewards, values, masks, cfg.gamma, cfg.lam)

        # ---- 更新 ----
        self.actor.train()
        self.critic.train()
        total_loss = torch.tensor(0.0, device=device)
        for _ in range(cfg.ppo_epochs):
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                new_sum, new_token = response_logprobs(self.actor, ids_t, labels_t, device)
                new_vals = self.critic(ids_t, mask_t)

            ratio = torch.exp(new_sum - old_sum)
            adv = (advantages * masks).sum(-1) / masks.sum(-1).clamp(min=1)
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

            # 逐 token value loss（只算响应位置）
            val_loss = 0.0
            for i in range(b):
                ln = resp_lens[i]
                val_loss = val_loss + nn.functional.mse_loss(new_vals[i, starts[i] - 1 : starts[i] - 1 + ln],
                                                            returns[i, :ln])
            val_loss = val_loss / b

            loss = policy_loss + cfg.vf_coef * val_loss
            total_loss = total_loss + loss.detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            self.actor_opt.step()
            self.critic_opt.step()

        self.step += 1
        return {
            "loss": float(total_loss.item()),
            "reward": float(rm_rewards.mean().item()),
            "kl": float(kl_mean.item()),
            "response_len": float(sum(resp_lens) / max(1, len(resp_lens))),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "step": self.step,
                "config": self.cfg,
            },
            path,
        )
