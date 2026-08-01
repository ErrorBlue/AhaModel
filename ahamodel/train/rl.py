"""RL 阶段共享工具：响应 log 概率、KL 散度、GAE。"""

from __future__ import annotations

from typing import Tuple

import torch


def response_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (每样本响应段 logp 之和, 逐 token logp (b,s))。

    labels: (b, s)，响应段位置为目标 token id，其余 -100。
    logits[t] 预测 input_ids[t+1]，因此用 labels[t] 作为目标。
    """
    out = model(input_ids)
    logp = out.logits.log_softmax(dim=-1)  # (b, s, V)
    mask = labels != -100
    valid = labels.clamp(min=0).unsqueeze(-1)
    token_lp = logp.gather(-1, valid).squeeze(-1)  # (b, s)
    token_lp = token_lp.masked_fill(~mask, 0.0)
    return token_lp.sum(-1), token_lp


def per_token_kl(logp_ref: torch.Tensor, logp: torch.Tensor, mode: str = "k2") -> torch.Tensor:
    """逐 token KL(π_ref || π)。mode: k1=diff, k2=exp(diff)-diff-1（默认，无偏估计）。

    教学提示：GRPO 论文常用 k3 = ρ·logρ - ρ + 1（ρ=π/π_ref）；
    小模型教学场景 k2 即可，公式可自行替换对比。
    """
    diff = logp_ref - logp
    if mode == "k1":
        return diff
    if mode == "k2":
        return torch.exp(diff) - diff - 1.0
    if mode == "k3":
        rho = torch.exp(diff)
        return rho * diff  # 近似 ρlogρ-ρ+1 的简化形式
    raise ValueError(f"未知 KL mode: {mode}")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE：returns = advantages + values；无效位置置 0。

    rewards/values/masks 形状 (b, T)，T 为响应最大长度。
    """
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[0], device=rewards.device)
    next_value = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
        next_value = values[:, t]
    returns = advantages + values
    advantages = advantages * masks
    returns = returns * masks
    return advantages, returns


def make_response_labels(
    input_ids: torch.Tensor,
    response_start: torch.Tensor,
    response_len: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """构造 RL 阶段 labels：响应 token 是目标，其余 -100。

    labels[t] 应为 input_ids[t+1]：响应占 [start, end)，目标位置 [start-1, end-1)。
    """
    b, s = input_ids.shape
    labels = torch.full((b, s), -100, dtype=torch.long, device=input_ids.device)
    for i in range(b):
        st = int(response_start[i])
        ln = int(response_len[i])
        if st >= 1 and ln > 0:
            labels[i, st - 1 : st - 1 + ln] = input_ids[i, st : st + ln]
    return labels
