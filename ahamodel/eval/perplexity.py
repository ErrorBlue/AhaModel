"""困惑度（PPL）评测。"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from ahamodel.data.dataset import PretrainDataset, collate_lm


def evaluate_ppl(
    model,
    tokenizer,
    jsonl_path: str,
    max_seq_len: int = 512,
    batch_size: int = 8,
    device: str = "cpu",
    max_samples: int | None = None,
) -> float:
    """在留出语料上算 PPL（按真实 token 数加权平均）。"""
    ds = PretrainDataset(tokenizer, jsonl_path, max_seq_len, max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_lm(b, tokenizer.pad_id or 0))
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], labels=batch["labels"])
            n_tokens = int((batch["labels"] != -100).sum().item())
            total_nll += float(out.loss.item()) * n_tokens
            total_tokens += n_tokens
    return math.exp(total_nll / max(1, total_tokens))
