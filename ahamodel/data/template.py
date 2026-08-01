"""聊天模板：messages -> token ids，以及 SFT/DPO/RL 需要的掩码。

约定模板（与 vLLM 导出端保持一致）：
  <|system|>系统内容<|user|>用户内容<|assistant|>回答内容<|eos|><|user|>...
loss 只计算 assistant 回答（含其后的 <|eos|>），其余位置 label = -100。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


class ChatTemplate:
    def __init__(self, tokenizer, max_seq_len: int | None = None):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def encode_messages(self, messages: Sequence[dict]) -> List[int]:
        """完整编码一轮对话（含末尾 assistant 的 <|eos|>）。"""
        ids: List[int] = []
        for m in messages:
            role = m.get("role", "user")
            marker = self.tok.encode(f"<|{role}|>")
            content = m.get("content") or ""
            ids.extend(marker)
            if content:
                ids.extend(self.tok.encode(content))
            if role == "assistant":
                ids.append(self.tok.eos_id)
        return self._truncate(ids)

    def encode_with_labels(self, messages: Sequence[dict]) -> Tuple[List[int], List[int]]:
        """返回 (input_ids, labels)。labels[t] 是位置 t 要预测的目标 token id；
        非 assistant 内容位置为 -100；末尾位置无目标。"""
        ids: List[int] = []
        assistant_mask: List[int] = []  # 1 表示该位置属于 assistant 内容（含其后 eos）
        for m in messages:
            role = m.get("role", "user")
            marker = self.tok.encode(f"<|{role}|>")
            content = m.get("content") or ""
            for part_ids, is_assistant in (
                (marker, False),
                (self.tok.encode(content) if content else [], role == "assistant"),
            ):
                ids.extend(part_ids)
                assistant_mask.extend([1 if is_assistant else 0] * len(part_ids))
            if role == "assistant":
                ids.append(self.tok.eos_id)
                assistant_mask.append(1)
        ids = self._truncate(ids)
        assistant_mask = assistant_mask[: len(ids)]
        labels = [-100] * len(ids)
        for j in range(len(ids) - 1):
            if assistant_mask[j + 1]:
                labels[j] = ids[j + 1]
        return ids, labels

    def encode_rollout_prompt(self, messages: Sequence[dict]) -> List[int]:
        """把对话编码到最后一个 assistant 标记为止（不含 eos），供 RL rollout 续写。
        要求最后一条消息是 content 为空的 assistant。"""
        ids: List[int] = []
        last = len(messages) - 1
        for idx, m in enumerate(messages):
            role = m.get("role", "user")
            content = m.get("content") or ""
            ids.extend(self.tok.encode(f"<|{role}|>"))
            if content:
                ids.extend(self.tok.encode(content))
            if role == "assistant" and idx < last:
                ids.append(self.tok.eos_id)
        return self._truncate(ids)

    def _truncate(self, ids: List[int]) -> List[int]:
        if self.max_seq_len and len(ids) > self.max_seq_len:
            return ids[: self.max_seq_len]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids)
