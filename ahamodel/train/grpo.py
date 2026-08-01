"""GRPO 从零实现：无 critic，组内 advantage 归一化 + 规则奖励（RLVR 思路）。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, List

import torch

from ahamodel.config import StageConfig
from ahamodel.data.math_qa import rule_reward
from ahamodel.model.generate import generate_batch
from ahamodel.model.model import AhaForCausalLM
from ahamodel.train.rl import make_response_labels, per_token_kl, response_logprobs


class GRPOTrainer:
    def __init__(
        self,
        actor: AhaForCausalLM,
        reference: AhaForCausalLM,
        cfg: StageConfig,
        device: str,
        logger=None,
    ):
        self.actor = actor.to(device)
        self.reference = reference.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.step = 0
        self.opt = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def step_once(self, prompts: List[dict], tokenizer) -> dict:
        """每个 prompt 采样 group_size 条，组内归一化 advantage 后更新一次。"""
        cfg = self.cfg
        device = self.device
        pad_id = tokenizer.pad_id or 0
        group = cfg.group_size
        # prompts: [{"prompt": str, "answer": str}]
        prompt_ids = [tokenizer.encode(p["prompt"]) for p in prompts]
        batched = [pid for pid in prompt_ids for _ in range(group)]
        responses = generate_batch(
            self.actor,
            tokenizer,
            batched,
            max_new_tokens=cfg.response_max_len,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            device=device,
        )

        # 奖励：规则函数（答案匹配）；组内归一化 advantage
        rewards = []
        for gi, pid in enumerate(prompt_ids):
            g_rewards = [
                rule_reward(prompts[gi], tokenizer.decode(r)) for r in responses[gi * group : (gi + 1) * group]
            ]
            gt = torch.tensor(g_rewards, dtype=torch.float32, device=device)
            adv = (gt - gt.mean()) / (gt.std() + 1e-8)
            rewards.append((g_rewards, adv))

        # 组装 batch：所有响应一起计算 logp
        full_ids, labels, masks, advs = [], [], [], []
        T = max((len(r) for r in responses), default=1)
        ids_t = torch.full((len(responses), max(len(p) for p in prompt_ids) + T), pad_id, dtype=torch.long, device=device)
        starts = [len(p) for p in prompt_ids for _ in range(group)]
        resp_lens = [len(r) for r in responses]
        labels_t = torch.full_like(ids_t, -100)
        for i, (pid, r) in enumerate(zip(batched, responses)):
            ids_t[i, : len(pid)] = torch.tensor(pid, dtype=torch.long)
            ids_t[i, len(pid) : len(pid) + len(r)] = torch.tensor(r, dtype=torch.long)
            if len(r) > 0:
                labels_t[i, len(pid) - 1 : len(pid) - 1 + len(r)] = ids_t[i, len(pid) : len(pid) + len(r)]
        # 逐样本 advantage（广播到该响应每个 token）
        per_sample_adv = torch.cat([a for _, a in rewards], dim=0)  # (group*n,)

        with torch.no_grad():
            ref_sum, ref_token = response_logprobs(self.reference, ids_t, labels_t, device)
            old_sum, old_token = response_logprobs(self.actor, ids_t, labels_t, device)

        # token 级掩码（响应位置）
        resp_mask = torch.zeros_like(labels_t, dtype=torch.bool)
        for i, (st, ln) in enumerate(zip(starts, resp_lens)):
            if ln > 0:
                resp_mask[i, st - 1 : st - 1 + ln] = True
        kl_token = per_token_kl(ref_token, old_token, mode="k2")

        self.actor.train()
        self.opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            new_sum, new_token = response_logprobs(self.actor, ids_t, labels_t, device)

        ratio = torch.exp(new_sum - old_sum)
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
        policy_loss = -(torch.min(ratio, clipped) * per_sample_adv).mean()
        kl_loss = (kl_token * resp_mask).sum(-1) / resp_mask.sum(-1).clamp(min=1)
        loss = policy_loss - cfg.kl_coef * kl_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.opt.step()
        self.step += 1

        mean_reward = torch.tensor([g for g, _ in rewards]).float().mean().item()
        return {
            "loss": float(loss.item()),
            "reward": float(mean_reward),
            "kl": float(kl_token[resp_mask].mean().item()),
            "response_len": float(sum(resp_lens) / max(1, len(resp_lens))),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "step": self.step, "config": self.cfg}, path)
