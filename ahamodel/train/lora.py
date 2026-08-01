"""LoRA 从零实现：可插拔低秩适配层 + 合并/解合并。"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """低秩增量：delta = (x @ A @ B) * (alpha / r)。"""

    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * (1.0 / r) ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """包装一个冻结的 nn.Linear，forward = 原线性 + LoRA 增量。

    merge(): 把增量写回 weight（服务端部署时零额外开销）。
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.original_weight = linear.weight.detach().clone()  # 冻结副本（合并后用于还原）
        self.weight = nn.Parameter(self.original_weight.clone(), requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.lora = LoRALayer(self.in_features, self.out_features, r, alpha, dropout)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # 合并后增量已写入 weight，不再叠加 LoRA 分支
            return F.linear(x, self.weight, self.bias)
        out = F.linear(x, self.weight, self.bias)
        return out + self.lora(x)

    def merge(self) -> None:
        with torch.no_grad():
            delta = (self.lora.lora_A @ self.lora.lora_B) * self.lora.scaling
            self.weight.copy_(self.original_weight + delta.T)
        self.merged = True

    def unmerge(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.original_weight)
        self.merged = False

    def lora_state_dict(self) -> dict:
        return {"A": self.lora.lora_A.detach().clone(), "B": self.lora.lora_B.detach().clone(),
                "alpha": self.lora.scaling * self.lora.r, "r": self.lora.r}


DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    targets: Optional[Sequence[str]] = None,
) -> List[str]:
    """把模型里的目标 Linear 原地替换为 LoRALinear，并冻结其余参数。"""
    targets = set(targets or DEFAULT_TARGETS)
    replaced: List[str] = []

    def walk(module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
                replaced.append(path)
            else:
                walk(child, path)

    walk(model, "")
    # 冻结所有参数，再解冻 LoRA 的 A/B
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora.lora_A.requires_grad_(True)
            m.lora.lora_B.requires_grad_(True)
    print(f"LoRA 已应用到 {len(replaced)} 个层: {replaced}")
    return replaced


def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def save_lora_state(model: nn.Module) -> dict:
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            state[name] = m.lora_state_dict()
    return state


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
