"""LoRA 测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("torch")
import torch

from ahamodel.config import ModelConfig
from ahamodel.model.model import AhaForCausalLM
from ahamodel.train.lora import apply_lora, count_trainable, merge_lora, unmerge_lora


def _model():
    return AhaForCausalLM(
        ModelConfig(vocab_size=300, d_model=64, n_layers=2, q_heads=4, kv_heads=2, intermediate_size=128, max_seq_len=64)
    )


def test_apply_freezes_and_counts():
    model = _model()
    apply_lora(model, r=4, alpha=8, dropout=0.0)
    n_train = count_trainable(model)
    assert n_train > 0
    assert n_train < model.num_params()
    # 除 LoRA A/B 外全部冻结
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad


def test_merge_unmerge_equivalence():
    torch.manual_seed(0)
    model = _model()
    apply_lora(model, r=4, alpha=8, dropout=0.0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 300, (2, 16))
    model.train()
    loss = model(x, labels=x.clone()).loss
    loss.backward()
    opt.step()
    with torch.no_grad():
        out_lora = model(x).logits
        merge_lora(model)
        out_merged = model(x).logits
        unmerge_lora(model)
        out_unmerged = model(x).logits
    assert torch.allclose(out_lora, out_merged, atol=1e-5)
    assert torch.allclose(out_lora, out_unmerged, atol=1e-5)
