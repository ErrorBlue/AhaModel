"""HF 导出测试（需要 tokenizers / transformers / safetensors）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

tokenizers = pytest.importorskip("tokenizers")
from ahamodel.config import ModelConfig
from ahamodel.export.tokenizer_hf import export_hf_tokenizer
from ahamodel.tokenizer import BPE, SPECIAL_TOKENS, Tokenizer


def _tok():
    corpus = ["你好世界", "今天天气不错", "机器学习很有趣", "让我们学习大模型"]
    return Tokenizer(BPE(SPECIAL_TOKENS, max_vocab_size=300).train(corpus, min_freq=1, verbose=False))


def test_hf_tokenizer_export(tmp_path):
    tok = _tok()
    out = tmp_path / "hf"
    export_hf_tokenizer(tok, out, model_max_length=768)
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()

    from transformers import PreTrainedTokenizerFast

    hf = PreTrainedTokenizerFast(tokenizer_file=str(out / "tokenizer.json"))
    # 特殊 token 一致性：id 与自写 tokenizer 相同
    assert hf.convert_tokens_to_ids("<|user|>") == tok.user_id
    assert hf.convert_tokens_to_ids("<|eos|>") == tok.eos_id
    text = "你好，世界！"
    our_ids = tok.encode(text)
    hf_ids = hf.encode(text, add_special_tokens=False)
    # 普通文本编码结果一致
    assert our_ids == hf_ids
    assert hf.decode(hf_ids) == tok.decode(our_ids)


def test_hf_model_export_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers")

    from ahamodel.export.to_hf import export_to_hf
    from ahamodel.model.model import AhaForCausalLM

    torch.manual_seed(0)
    tok = _tok()
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, d_model=64, n_layers=2, q_heads=4, kv_heads=2,
        intermediate_size=128, max_seq_len=64,
    )
    model = AhaForCausalLM(cfg).eval()
    out = tmp_path / "hf_model"
    export_to_hf(model, tok, cfg, out)

    hf_model = transformers.LlamaForCausalLM.from_pretrained(out)
    hf_model.eval()
    ids = torch.tensor([[tok.encode("你好世界")]], dtype=torch.long)
    with torch.no_grad():
        ours = model(ids).logits
        theirs = hf_model(ids).logits
    assert torch.allclose(ours, theirs, atol=1e-4), (ours - theirs).abs().max()
