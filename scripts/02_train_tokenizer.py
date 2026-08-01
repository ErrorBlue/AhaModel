"""步骤 2：训练字节级 BPE tokenizer（语料来自 scripts/01_prepare_data.py）。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahamodel.tokenizer import BPE, SPECIAL_TOKENS, Tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练 BPE tokenizer")
    parser.add_argument("--corpus", default="data/pretrain.jsonl", help="清洗后的预训练语料（{text: str}）")
    parser.add_argument("--out", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument("--sample", type=int, default=20000, help="用于训练的样本条数（教学用子集）")
    parser.add_argument("--export-hf", default=None, help="可选：导出 HF tokenizer 到该目录")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        sys.exit(f"找不到语料 {corpus}，请先运行 scripts/01_prepare_data.py")

    texts = []
    with open(corpus, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.sample:
                break
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    print(f"使用 {len(texts)} 条文本训练 BPE（词表目标 {args.vocab_size}）")

    bpe = BPE(SPECIAL_TOKENS, max_vocab_size=args.vocab_size).train(texts)
    tok = Tokenizer(bpe)
    tok.save(args.out)

    sample = "你好，世界！Hello, world! 123"
    ids = tok.encode(sample)
    print(f"词表大小: {tok.vocab_size}（目标 {args.vocab_size}）")
    print(f"示例编码: {sample} -> {ids[:30]}...")
    print(f"示例解码: {tok.decode(ids)}")
    print(f"tokenizer 已保存: {args.out}")

    if args.export_hf:
        from ahamodel.export.tokenizer_hf import export_hf_tokenizer

        export_hf_tokenizer(tok, args.export_hf)


if __name__ == "__main__":
    main()
