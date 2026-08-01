"""RL 阶段模型对比：用 RM 打分，输出偏好胜率与平均奖励。"""

from __future__ import annotations

from typing import List, Optional

import torch

from ahamodel.data.template import ChatTemplate
from ahamodel.model.generate import generate_batch


def _score_responses(model, rm, tokenizer, prompts_ids, max_new_tokens, device, temperature, top_p) -> List[float]:
    """对每条 prompt 生成一条回答，再用 RM 打分。"""
    responses = generate_batch(
        model, tokenizer, prompts_ids,
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        top_k=0, device=device,
    )
    scores = []
    for pid, resp in zip(prompts_ids, responses):
        if not resp:
            scores.append(0.0)
            continue
        full = pid + resp
        ids_t = torch.tensor([full], dtype=torch.long, device=device)
        start = torch.tensor([len(pid)], dtype=torch.long, device=device)
        mask = torch.ones_like(ids_t)
        with torch.no_grad():
            scores.append(float(rm(ids_t, start, mask).item()))
    return scores


def compare_models(
    sft_model,
    rl_model,
    rm,
    tokenizer,
    dpo_items: List[dict],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cpu",
) -> dict:
    """对比 SFT 与 RL 模型：生成质量用 RM 打分，输出胜率。"""
    template = ChatTemplate(tokenizer)
    prompts_ids = []
    for item in dpo_items:
        user_msgs = [m for m in item["chosen"] if m["role"] == "user"]
        prompts_ids.append(template.encode_rollout_prompt(user_msgs + [{"role": "assistant", "content": ""}]))

    sft_model.eval()
    rl_model.eval()
    sft_scores = _score_responses(sft_model, rm, tokenizer, prompts_ids, max_new_tokens, device, temperature, top_p)
    rl_scores = _score_responses(rl_model, rm, tokenizer, prompts_ids, max_new_tokens, device, temperature, top_p)
    wins = sum(1 for a, b in zip(rl_scores, sft_scores) if a >= b)
    result = {
        "win_rate_rl_vs_sft": wins / max(1, len(sft_scores)),
        "mean_rm_reward_sft": sum(sft_scores) / max(1, len(sft_scores)),
        "mean_rm_reward_rl": sum(rl_scores) / max(1, len(rl_scores)),
        "n_items": len(sft_scores),
    }
    print(
        f"RL vs SFT 胜率: {result['win_rate_rl_vs_sft']:.2%}  "
        f"平均 RM 奖励: SFT {result['mean_rm_reward_sft']:.4f} / RL {result['mean_rm_reward_rl']:.4f}"
    )
    return result
