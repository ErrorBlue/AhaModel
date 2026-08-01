"""把自写 BPE tokenizer 导出为 HuggingFace tokenizer.json（vLLM / transformers 可用）。"""

from __future__ import annotations

import json
from pathlib import Path


def _byte_encoder() -> dict:
    """GPT-2 风格字节映射：tokenizers 库 ByteLevel 使用同一张表。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def export_hf_tokenizer(tokenizer, out_dir: str | Path, model_max_length: int = 768) -> Path:
    """导出 tokenizer.json + tokenizer_config.json + special_tokens_map.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    byte_map = _byte_encoder()

    # 词表：特殊 token 用我们自己的 id；字节 token 与合并 token 用 GPT-2 字符表示
    vocab = dict(tokenizer.special_to_id)
    for b in range(256):
        vocab[byte_map[b]] = tokenizer.bpe.byte_start + b
    for a, b, new in tokenizer.bpe.merges:
        s = "".join(byte_map[x] for x in tokenizer.bpe.id_to_bytes[new])
        vocab[s] = new

    merges = []
    for a, b, _new in tokenizer.bpe.merges:
        sa = "".join(byte_map[x] for x in tokenizer.bpe.id_to_bytes[a])
        sb = "".join(byte_map[x] for x in tokenizer.bpe.id_to_bytes[b])
        merges.append(f"{sa} {sb}")

    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, pre_tokenizers

    hf = HFTokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<|unk|>"))
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    for s in tokenizer.special_tokens:
        hf.add_special_tokens([s])
    hf.save(str(out_dir / "tokenizer.json"))

    chat_template = (
        "{% for message in messages %}{{ '<|' + message['role'] + '|>' }}"
        "{{ message['content'] }}{% if message['role'] == 'assistant' %}<|eos|>{% endif %}"
        "{% endfor %}"
    )
    tok_config = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "add_bos_token": False,
        "add_eos_token": False,
        "model_max_length": model_max_length,
        "chat_template": chat_template,
    }
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_config, f, ensure_ascii=False, indent=2)

    special_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
    }
    with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_map, f, ensure_ascii=False, indent=2)

    print(f"tokenizer 已导出到 {out_dir}")
    return out_dir
