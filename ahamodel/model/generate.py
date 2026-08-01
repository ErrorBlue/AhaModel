"""自回归生成：贪心 / top-k / top-p 采样，支持单条与批量（RL rollout 用）。"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """top-k 与 top-p 过滤（nucleus sampling）。top_k<=0 表示不启用。"""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        logits = logits.masked_fill(logits < topk_vals[..., -1:], filter_value)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
        remove_mask = cum_probs - torch.softmax(sorted_logits.float(), dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, filter_value)
        logits = torch.scatter(sorted_logits, -1, sorted_indices, sorted_logits)  # 还原顺序
        # scatter 的简便写法：把 sorted 结果放回原索引
        out = torch.full_like(logits, filter_value)
        out.scatter_(-1, sorted_indices, sorted_logits)
        logits = out
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> torch.Tensor:
    """输入最后一位置的 logits (b, vocab)，返回采样结果 (b, 1)。"""
    if logits.dim() == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)  # 兼容 (b, 1, V) 输入
    logits = logits.float() / max(temperature, 1e-5)
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
    eos_id: Optional[int] = None,
    device: str = "cpu",
    use_cache: bool = True,
) -> List[int]:
    """单条生成：返回完整 token 序列（含 prompt）。"""
    model.eval()
    eos_id = eos_id if eos_id is not None else tokenizer.eos_id
    max_pos = getattr(getattr(model, "cfg", None), "max_seq_len", None)
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    past = None
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if max_pos is not None and len(generated) >= max_pos:
                break  # 达到模型最大位置，提前停止
            out = model(input_ids, past_key_values=past, use_cache=use_cache)
            next_id = sample_next_token(
                out.logits[:, -1:, :], temperature, top_k, top_p, do_sample
            )
            nid = next_id.item()
            generated.append(nid)
            if nid == eos_id:
                break
            past = out.past_key_values
            input_ids = next_id
    return generated


def generate_batch(
    model,
    tokenizer,
    prompts: List[Sequence[int]],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_id: Optional[int] = None,
    device: str = "cpu",
) -> List[List[int]]:
    """批量生成（左 padding），返回每条「新增」的 token 序列（不含 prompt）。"""
    model.eval()
    eos_id = eos_id if eos_id is not None else tokenizer.eos_id
    max_pos = getattr(getattr(model, "cfg", None), "max_seq_len", None)
    batch = len(prompts)
    max_plen = max(len(p) for p in prompts)
    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0

    padded = torch.full((batch, max_plen), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((batch, max_plen), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        lp = max_plen - len(p)
        padded[i, lp:] = torch.tensor(p, dtype=torch.long)
        mask[i, lp:] = 1
    # 左 padding：真实位置 = arange - 左补位数（补位处为负，但其注意力被 mask 屏蔽）
    left_pad = (mask == 0).sum(dim=1)  # (b,)
    pos = torch.arange(max_plen, device=device).unsqueeze(0).expand(batch, -1) - left_pad.unsqueeze(1)

    # 每行当前真实生成位置（RoPE 位置 id 用）
    cur_pos = torch.tensor([len(p) for p in prompts], dtype=torch.long, device=device)
    unfinished = torch.ones(batch, dtype=torch.bool, device=device)
    responses: List[List[int]] = [[] for _ in range(batch)]
    past = None
    input_ids = padded

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past is None:
                out = model(input_ids, past_key_values=None, use_cache=True,
                            attention_mask=mask, position_ids=pos)
            else:
                # 生成步：每行 1 个新 token
                out = model(input_ids, past_key_values=past, use_cache=True,
                            attention_mask=torch.ones((batch, 1), dtype=torch.long, device=device),
                            position_ids=cur_pos.unsqueeze(1))
            next_ids = sample_next_token(
                out.logits[:, -1:, :], temperature, top_k, top_p, do_sample=True
            )  # (b, 1)
            nids = next_ids.squeeze(1)
            for i in range(batch):
                if unfinished[i]:
                    responses[i].append(int(nids[i]))
                    if int(nids[i]) == eos_id:
                        unfinished[i] = False
                    if max_pos is not None and cur_pos[i].item() >= max_pos - 1:
                        unfinished[i] = False  # 超出模型位置上限
            if not unfinished.any():
                break
            past = out.past_key_values
            # 已结束的行继续喂 pad 占位（logits 会被忽略）
            input_ids = torch.where(unfinished.unsqueeze(1), next_ids,
                                    torch.full_like(next_ids, pad_id))
            cur_pos = cur_pos + unfinished.long()
    return responses
