"""SwiGLU 前馈网络（HF Llama 同款命名）。"""

from __future__ import annotations

import torch.nn as nn

from ahamodel.config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)

    def forward(self, x):
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
