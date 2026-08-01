"""奖励模型：transformer 骨干 + 响应段平均池化 -> 标量。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ahamodel.config import ModelConfig
from ahamodel.model.model import AhaTransformer


class RewardModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AhaTransformer(cfg)
        self.reward_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        response_start: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s = input_ids.shape
        position_ids = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)
        hidden, _ = self.model(input_ids, position_ids, use_cache=False, attention_mask=attention_mask)

        # 只平均「响应段」的 hidden，避免 prompt 长度干扰奖励尺度
        idx = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)
        in_response = idx >= response_start.unsqueeze(1)
        if attention_mask is not None:
            in_response = in_response & (attention_mask == 1)
        counts = in_response.sum(1).clamp(min=1).unsqueeze(1)
        pooled = (hidden * in_response.unsqueeze(-1).to(hidden.dtype)).sum(1) / counts
        return self.reward_head(pooled).squeeze(-1)  # (b,)
