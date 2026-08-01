"""RoPE 旋转位置编码（实现与 HF Llama 完全一致，便于导出对齐）。"""

from __future__ import annotations

import torch


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 cos/sin 表。

    返回两个 (max_seq_len, head_dim) 张量：偶数位/奇数位分别对应同一频率，
    即 HF 中 `torch.cat((freqs, freqs), dim=-1)` 后取 cos/sin 的形式。
    """
    assert head_dim % 2 == 0, "head_dim 必须为偶数"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin: (seq, head_dim) 或 (batch, seq, head_dim)，与 q/k 的最后一维对齐。"""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
