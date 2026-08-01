"""文本清洗：HTML 去除、空白归一化、质量过滤、去重。"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Iterable, Optional

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """单条文本清洗。"""
    if not text:
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = CTRL_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
    return cjk / len(text)


def quality_filter(
    text: str,
    min_len: int = 50,
    max_len: int = 10000,
    min_cjk_ratio: float = 0.0,
) -> bool:
    """质量过滤：长度区间 + 中文占比下限（可配置为 0 关闭）。"""
    n = len(text)
    if n < min_len or n > max_len:
        return False
    if min_cjk_ratio > 0 and cjk_ratio(text) < min_cjk_ratio:
        return False
    return True


def dedup_texts(texts: Iterable[str]) -> Iterable[str]:
    """基于全文 sha256 去重（内存占用小，适合流式处理）。"""
    seen: set[str] = set()
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        yield text


def clean_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    min_len: int = 50,
    max_len: int = 10000,
    min_cjk_ratio: float = 0.0,
    do_dedup: bool = True,
    field: str = "text",
) -> tuple[int, int]:
    """流式清洗 {field: str} 格式的 jsonl，返回 (输入条数, 保留条数)。"""
    import json

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    seen: set[str] = set()
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            text = clean_text(item.get(field, ""))
            if not quality_filter(text, min_len, max_len, min_cjk_ratio):
                continue
            if do_dedup:
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
            fout.write(json.dumps({field: text}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"清洗完成: {total} -> {kept} 条")
    return total, kept
