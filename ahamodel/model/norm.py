"""RMSNorm：LLaMA 风格归一化。"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 fp32 计算方差避免低精度下不稳定，再乘回原精度
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight).to(x.dtype)
