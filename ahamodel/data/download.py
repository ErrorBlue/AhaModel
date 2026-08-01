"""数据下载：minimind 开源数据集（HF 优先，支持镜像/ModelScope 提示）+ 维基中文教学路径。"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Optional

MINIMIND_DATASET = "jingyaogong/minimind_dataset"

# 各阶段默认文件（minimind-3 主线 mini 系列）
FILES = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "sft": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
    "rlaif": "rlaif.jsonl",
}


def download_minimind_file(name: str, out_dir: str | Path, max_lines: Optional[int] = None, seed: int = 42) -> Path:
    """下载 minimind 数据集指定文件并（可选）采样到 out_dir/<name>.jsonl。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = FILES.get(name, name)
    dst = out_dir / f"{name}.jsonl"
    if dst.exists():
        print(f"已存在 {dst}，跳过下载")
        return dst

    try:
        from huggingface_hub import hf_hub_download

        raw = hf_hub_download(
            repo_id=MINIMIND_DATASET,
            filename=filename,
            repo_type="dataset",
        )
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 下载失败。国内服务器可设置镜像：\n"
            "  export HF_ENDPOINT=https://hf-mirror.com\n"
            "或从 ModelScope 手动下载（https://www.modelscope.cn/datasets/gongjy/minimind_dataset）：\n"
            f"  文件 {filename} 放到 {out_dir}/ 下并改名为 {name}.jsonl"
        ) from e

    if max_lines:
        sample_jsonl(Path(raw), dst, max_lines, seed)
    else:
        import shutil

        shutil.copyfile(raw, dst)
    print(f"数据就绪: {dst}")
    return dst


def sample_jsonl(src: str | Path, dst: str | Path, n: int, seed: int = 42) -> None:
    """从 jsonl 均匀采样 n 行（顺序读取，控制内存）。"""
    rng = random.Random(seed)
    keep: list[dict] = []
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if len(keep) < n:
                keep.append(item)
            else:
                j = rng.randint(0, i)
                if j < n:
                    keep[j] = item
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for item in keep:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"采样 {len(keep)} 行 -> {dst}")


def sample_jsonl_head(src: str | Path, dst: str | Path, n: int) -> None:
    """取前 n 行（教学用：保持数据原始顺序）。"""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if count >= n:
                break
            line = line.strip()
            if line:
                fout.write(line + "\n")
                count += 1
    print(f"取前 {count} 行 -> {dst}")


def tail_jsonl(src: str | Path, dst: str | Path, n: int) -> None:
    """取最后 n 行作为留出集（与训练文件不重叠）。"""
    tail: deque = deque(maxlen=n)
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tail.append(line)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(tail) + "\n")
    print(f"留出 {len(tail)} 行 -> {dst}")


def split_eval_set(pretrain_path: str | Path, eval_path: str | Path, n: int = 1000) -> None:
    """预训练数据下载后自动切出 PPL 评测留出集。"""
    if Path(eval_path).exists():
        print(f"已存在 {eval_path}，跳过")
        return
    tail_jsonl(pretrain_path, eval_path, n)


def download_wiki_zh(n: int = 10000, out_path: str | Path = "data/pretrain_wiki.jsonl") -> Path:
    """教学路径：从 HF 拉中文 Wikipedia 子集并清洗，产出 {"text": ...} jsonl。"""
    from datasets import load_dataset

    from ahamodel.data.clean import clean_text, quality_filter

    out_path = Path(out_path)
    if out_path.exists():
        print(f"已存在 {out_path}，跳过")
        return out_path
    print(f"下载中文 Wikipedia 前 {n} 篇……")
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split=f"train[:{n}]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = clean_text(row["text"])
            if quality_filter(text):
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"清洗后保留 {kept}/{n} 篇 -> {out_path}")
    return out_path
