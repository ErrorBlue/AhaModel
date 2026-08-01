"""步骤 2：下载并准备各阶段数据（minimind 开源数据集 + 维基中文清洗教学路径）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahamodel.data.download import download_minimind_file, download_wiki_zh, split_eval_set


def main():
    parser = argparse.ArgumentParser(description="准备全阶段数据")
    parser.add_argument(
        "--stage",
        choices=["pretrain", "sft", "dpo", "rlaif", "wiki", "all"],
        default="all",
        help="all = pretrain+sft+dpo+rlaif",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-lines", type=int, default=None, help="每个文件采样行数（smoke 用）")
    parser.add_argument("--wiki-n", type=int, default=10000, help="维基清洗路径的篇数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stages = ["pretrain", "sft", "dpo", "rlaif"] if args.stage == "all" else [args.stage]
    for s in stages:
        if s == "wiki":
            download_wiki_zh(n=args.wiki_n, out_path=data_dir / "pretrain_wiki.jsonl")
        else:
            download_minimind_file(s, data_dir, max_lines=args.max_lines)
        if s == "pretrain":
            # 自动留出 1000 行做 PPL 评测（与训练文件不重叠）
            split_eval_set(data_dir / "pretrain.jsonl", data_dir / "pretrain_eval.jsonl", n=1000)
    print("数据准备完成。教学路径（自制语料）见 docs/03-数据清洗.md")


if __name__ == "__main__":
    main()
