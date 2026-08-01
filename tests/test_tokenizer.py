"""BPE / Tokenizer 纯 Python 测试（不需要 torch）。"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahamodel.tokenizer import BPE, SPECIAL_TOKENS, Tokenizer


def _make_corpus(n=200, seed=0):
    rng = random.Random(seed)
    return [
        "".join(rng.choice("abcde 中文测试12345") for _ in range(rng.randint(1, 60)))
        for _ in range(n)
    ]


def _naive_apply(bpe, ids):
    """朴素 BPE 编码：每轮全量找最小 rank pair，慢但显然正确。"""
    ids = list(ids)
    while True:
        best = None
        for i in range(len(ids) - 1):
            r = bpe.merge_rank.get((ids[i], ids[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        _, i = best
        ids[i] = bpe.merge_map[(ids[i], ids[i + 1])]
        del ids[i + 1]
    return ids


def test_bpe_vocab_size_and_merge_count():
    bpe = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(_make_corpus(), verbose=False)
    assert bpe.vocab_size == 300
    assert bpe.num_specials == len(SPECIAL_TOKENS)
    assert len(bpe.merges) == 300 - len(SPECIAL_TOKENS) - 256
    # 特殊 token 占用前 7 个 id，字节占用其后 256 个
    assert bpe.byte_start == len(SPECIAL_TOKENS)
    assert bpe.byte_end == len(SPECIAL_TOKENS) + 256


def test_bpe_fast_matches_naive():
    bpe = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(_make_corpus(), verbose=False)
    corpus = _make_corpus(50)
    for text in corpus:
        fast = bpe._apply_merges(bpe.encode_bytes(text.encode("utf-8")))
        slow = _naive_apply(bpe, bpe.encode_bytes(text.encode("utf-8")))
        assert fast == slow


def test_roundtrip_and_unicode():
    bpe = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(_make_corpus(), verbose=False)
    tok = Tokenizer(bpe)
    samples = [
        "你好，世界！Hello, world! 123",
        "emoji test \U0001F680",
        "a" * 50,
        "",
        "  spaces  and\t tabs ",
        "<|user|>特殊token<|assistant|>",
        "中文中文中文" * 10,
    ]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s


def test_deterministic_training():
    corpus = _make_corpus()
    b1 = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(corpus, verbose=False)
    b2 = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(corpus, verbose=False)
    assert b1.merges == b2.merges


def test_special_tokens_encoding():
    bpe = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(_make_corpus(), verbose=False)
    tok = Tokenizer(bpe)
    assert tok.encode("<|pad|>") == [0]
    assert tok.encode("<|eos|>") == [tok.eos_id]
    assert tok.pad_id == 0 and tok.eos_id == 3 and tok.user_id == 5 and tok.assistant_id == 6


def test_save_load_roundtrip(tmp_path):
    bpe = BPE(SPECIAL_TOKENS, max_vocab_size=300).train(_make_corpus(), verbose=False)
    tok = Tokenizer(bpe)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    tok2 = Tokenizer.load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.bpe.merges == tok.bpe.merges
    for s in ["你好世界", "测试 test 123"]:
        assert tok2.decode(tok2.encode(s)) == tok.decode(tok.encode(s)) == s
