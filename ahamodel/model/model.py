"""Decoder-only Transformer：结构对齐 HF Llama，导出零映射。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ahamodel.config import ModelConfig
from ahamodel.model.attention import Attention
from ahamodel.model.mlp import MLP
from ahamodel.model.norm import RMSNorm


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    hidden_states: Optional[torch.Tensor] = None  # 最后一层输出，供 RM/Critic 使用


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, position_ids, past_key_value=None, use_cache=False, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        attn_out, present = self.self_attn(
            x,
            position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, present


class AhaTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=False, attention_mask=None):
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            # padding 位置直接清零，避免其携带的任意值参与后续计算
            x = x * attention_mask[..., None].to(x.dtype)
        presents: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, position_ids, past, use_cache, attention_mask)
            presents.append(present)
        x = self.norm(x)
        return x, presents


class AhaForCausalLM(nn.Module):
    """因果语言模型：embed_tokens + layers + norm + lm_head。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AhaTransformer(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> CausalLMOutput:
        b, s = input_ids.shape
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(past_len, past_len + s, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(b, -1)

        hidden, presents = self.model(
            input_ids,
            position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            # 只对 label != -100 的位置算交叉熵（SFT/RL 阶段用于掩码）；
            # 全被忽略时交叉熵会得到 NaN（0/0），这里显式返回 0
            valid = labels != -100
            if valid.any():
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = torch.zeros((), device=logits.device)
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=presents if use_cache else None,
            hidden_states=hidden if output_hidden_states else None,
        )

    def num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
