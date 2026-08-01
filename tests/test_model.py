"""模型结构 / KV Cache / 生成测试（需要 torch）。"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("torch")
import torch

from ahamodel.config import ModelConfig
from ahamodel.model.generate import generate, generate_batch
from ahamodel.model.model import AhaForCausalLM


def _cfg(**kw):
    defaults = dict(
        vocab_size=300, d_model=64, n_layers=2, q_heads=4, kv_heads=2,
        intermediate_size=128, max_seq_len=64,
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


def test_forward_backward():
    torch.manual_seed(0)
    model = AhaForCausalLM(_cfg())
    x = torch.randint(0, 300, (2, 32))
    out = model(x, labels=x.clone())
    assert out.logits.shape == (2, 32, 300)
    assert out.loss is not None
    out.loss.backward()


def test_kv_cache_matches_no_cache():
    torch.manual_seed(0)
    model = AhaForCausalLM(_cfg()).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        ids_c, seq, past = [], prompt, None
        for _ in range(6):
            out = model(seq, past_key_values=past, use_cache=True)
            nid = out.logits[:, -1].argmax()
            ids_c.append(int(nid))
            past = out.past_key_values
            seq = nid.view(1, 1)
        ids_n, seq = [], prompt
        for _ in range(6):
            out = model(seq)
            nid = out.logits[:, -1].argmax()
            ids_n.append(int(nid))
            seq = torch.cat([seq, nid.view(1, 1)], dim=1)
    assert ids_c == ids_n


def test_padding_attention_no_nan():
    torch.manual_seed(0)
    model = AhaForCausalLM(_cfg()).eval()
    padded = torch.tensor([[1, 2, 3], [0, 4, 5]])
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    pos = torch.tensor([[0, 1, 2], [-1, 0, 1]])
    with torch.no_grad():
        out = model(padded, attention_mask=mask, position_ids=pos)
    assert not torch.isnan(out.logits).any()


def test_generate_and_batch():
    torch.manual_seed(0)
    model = AhaForCausalLM(_cfg()).eval()
    tok = SimpleNamespace(eos_id=3, pad_id=0)
    gen = generate(model, tok, [1, 2, 3], max_new_tokens=8, device="cpu")
    assert len(gen) > 5
    genb = generate_batch(model, tok, [[1, 2], [3, 4, 5]], max_new_tokens=5, device="cpu")
    assert len(genb) == 2
    assert all(len(r) <= 5 for r in genb)


def test_attention_shapes_gqa():
    torch.manual_seed(0)
    model = AhaForCausalLM(_cfg(q_heads=8, kv_heads=2, d_model=64))
    x = torch.randint(0, 300, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 300)
