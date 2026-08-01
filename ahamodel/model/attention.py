"""因果自注意力：GQA + KV Cache。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ahamodel.config import ModelConfig
from ahamodel.model.rope import apply_rotary_pos_emb, precompute_freqs_cis


def repeat_kv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_rep: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GQA：把 kv 头复制 n_rep 份对齐 q 的头数（教学起见用 repeat，注释说明 expand 等价）。"""
    if n_rep == 1:
        return q, k, v
    b, kv_heads, s, head_dim = k.shape
    k = k[:, :, None, :, :].expand(b, kv_heads, n_rep, s, head_dim).reshape(b, kv_heads * n_rep, s, head_dim)
    v = v[:, :, None, :, :].expand(b, kv_heads, n_rep, s, head_dim).reshape(b, kv_heads * n_rep, s, head_dim)
    return q, k, v


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim

        # 命名与 HF Llama 对齐（导出时零映射）
        self.q_proj = nn.Linear(cfg.d_model, cfg.q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_heads * self.head_dim, cfg.d_model, bias=False)

        cos, sin = precompute_freqs_cis(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)

        cos = self.cos[position_ids].unsqueeze(1)  # (b, 1, s, head_dim)
        sin = self.sin[position_ids].unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kv_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            kv_len = pk.shape[2]

        # 缓存必须按 kv_heads 维度保存（repeat_kv 复制之后再存会头数不匹配）
        present = (k, v) if use_cache else None
        q, k, v = repeat_kv(q, k, v, self.q_heads // self.kv_heads)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (b, h, s_q, s_kv)
        # 因果掩码：diagonal = kv_len + 1 时，掩掉所有“未来位置”
        causal = torch.triu(
            torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=x.device),
            diagonal=kv_len + 1,
        )
        attn = attn.masked_fill(causal, float("-inf"))
        if attention_mask is not None:
            # attention_mask: (b, s) 1=保留 0=padding；expand 后与因果掩码合并
            pad = attention_mask[:, None, None, :] == 0  # (b,1,1,s_kv)
            attn = attn.masked_fill(pad.expand(b, q.shape[1], q.shape[2], k.shape[2]), float("-inf"))

        probs = torch.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        # 全被 mask 的行（如左 padding 的查询位）softmax 会产生 NaN；
        # 0 * NaN = NaN 会污染输出，这里统一归零（对应行本身也不会被使用）
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.attn_dropout(probs)
        out = probs @ v  # (b, h, s_q, head_dim)
        out = out.transpose(1, 2).reshape(b, s, -1)
        out = self.o_proj(out)

        return out, present
