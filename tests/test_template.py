"""聊天模板掩码测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahamodel.data.template import ChatTemplate
from ahamodel.tokenizer import BPE, SPECIAL_TOKENS, Tokenizer


def _tok():
    corpus = ["你好世界", "今天天气不错", "机器学习很有趣", "让我们学习大模型", "回答这个问题"]
    return Tokenizer(BPE(SPECIAL_TOKENS, max_vocab_size=300).train(corpus, min_freq=1, verbose=False))


def test_labels_only_on_assistant():
    tok = _tok()
    ids, labels = ChatTemplate(tok).encode_with_labels(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
        ]
    )
    assert ids[-1] == tok.eos_id and labels[-1] == -100
    # 每个真实标签都是下一位置 token 的 id（teacher forcing 移位）
    for j in range(len(ids) - 1):
        if labels[j] != -100:
            assert labels[j] == ids[j + 1]
    n_real = sum(1 for v in labels if v != -100)
    assert n_real > 0
    # 标签数应等于 assistant 内容 + eos 的 token 数
    asst_len = len(tok.encode("你好！有什么可以帮你？"))
    assert n_real == asst_len + 1


def test_rollout_prompt_ends_with_assistant_marker():
    tok = _tok()
    ids = ChatTemplate(tok).encode_rollout_prompt(
        [{"role": "user", "content": "问题"}, {"role": "assistant", "content": ""}]
    )
    assert ids[-1] == tok.assistant_id
    assert tok.eos_id not in ids


def test_multiturn_masking():
    tok = _tok()
    ids, labels = ChatTemplate(tok).encode_with_labels(
        [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ]
    )
    n_real = sum(1 for v in labels if v != -100)
    assert n_real == len(tok.encode("第一答")) + 1 + len(tok.encode("第二答")) + 1
