"""Tokenizer 封装：特殊 token 处理 + 编解码 + 存取 + 语料统计。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ahamodel.tokenizer.bpe import BPE

# 特殊 token 顺序即 id 顺序，训练模板与导出都要保持一致
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class Tokenizer:
    """特殊 token 感知的编解码器。"""

    def __init__(self, bpe: BPE):
        self.bpe = bpe
        self.special_tokens: List[str] = bpe.special_tokens
        self.special_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self.pad_id = self.special_to_id.get("<|pad|>")
        self.unk_id = self.special_to_id.get("<|unk|>")
        self.bos_id = self.special_to_id.get("<|bos|>")
        self.eos_id = self.special_to_id.get("<|eos|>")
        self.system_id = self.special_to_id.get("<|system|>")
        self.user_id = self.special_to_id.get("<|user|>")
        self.assistant_id = self.special_to_id.get("<|assistant|>")
        self.vocab_size = bpe.vocab_size

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def _split_specials(self, text: str) -> List[Tuple[bool, str]]:
        """把文本按特殊 token 子串切分，返回 [(是否特殊, 片段)]。"""
        parts: List[Tuple[bool, str]] = []
        rest = text
        while rest:
            best_pos = len(rest)
            best_tok: Optional[str] = None
            for s in self.special_tokens:
                pos = rest.find(s)
                if pos != -1 and pos < best_pos:
                    best_pos = pos
                    best_tok = s
            if best_tok is None:
                parts.append((False, rest))
                break
            if best_pos > 0:
                parts.append((False, rest[:best_pos]))
            parts.append((True, best_tok))
            rest = rest[best_pos + len(best_tok):]
        return parts

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos and self.bos_id is not None:
            ids.append(self.bos_id)
        for is_special, piece in self._split_specials(text):
            if is_special:
                ids.append(self.special_to_id[piece])
            else:
                ids.extend(self.bpe.encode_bytes(piece.encode("utf-8")))
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(self, ids: Sequence[int]) -> str:
        chunks: List[str] = []
        buf = bytearray()
        for i in ids:
            if i < self.bpe.num_specials:
                if buf:
                    chunks.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                chunks.append(self.special_tokens[i])
            else:
                buf.extend(self.bpe.id_to_bytes.get(i, b""))
        if buf:
            chunks.append(buf.decode("utf-8", errors="replace"))
        return "".join(chunks)

    # ------------------------------------------------------------------
    # 语料统计 / 训练辅助
    # ------------------------------------------------------------------
    def count_tokens(self, texts: Iterable[str]) -> int:
        return sum(len(self.encode(t)) for t in texts)

    def encode_stream(self, texts: Iterable[str]) -> Iterable[List[int]]:
        for t in texts:
            yield self.encode(t)

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"bpe": self.bpe.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        return cls(BPE.from_dict(d["bpe"]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
