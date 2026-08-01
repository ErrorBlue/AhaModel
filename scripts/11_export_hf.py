"""步骤 11：导出 HF 格式（config.json + safetensors + tokenizer.json）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ahamodel.config import load_model_config
from ahamodel.export.to_hf import export_to_hf
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.utils.cli import infer_model_config


def main():
    parser = argparse.ArgumentParser(description="导出 HF 格式")
    parser.add_argument("--model-path", required=True, help="训练 checkpoint（.pt）")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--out-dir", default="checkpoints/hf")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    mcfg = infer_model_config(args.model_path, load_model_config(args.model_config), device)
    tok = Tokenizer.load(args.tokenizer)
    mcfg.vocab_size = tok.vocab_size
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    # 兼容 LoRA 合并前保存的权重：丢弃 LoRA 结构键
    state = {k: v for k, v in state.items() if ".lora." not in k and "original_weight" not in k}
    model = AhaForCausalLM(mcfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"加载权重: {args.model_path}（参数量 {model.num_params()/1e6:.2f}M）")
    export_to_hf(model, tok, mcfg, args.out_dir)


if __name__ == "__main__":
    main()
