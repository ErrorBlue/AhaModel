"""步骤 5：LoRA 微调（从零实现，支持合并导出）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from ahamodel.data.dataset import SftDataset, collate_sft
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.lora import apply_lora, count_trainable, merge_lora
from ahamodel.train.trainer import Trainer
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, setup_stage


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="基础模型 checkpoint（SFT/预训练）")
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/lora.yaml", ROOT)
    model_path = args.model_path or cfg.model_path
    mcfg = infer_model_config(model_path, mcfg, device)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size
    model = AhaForCausalLM(mcfg)
    load_pretrained(model, model_path, device)
    apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, targets=cfg.lora_targets)
    print(f"LoRA 可训练参数: {count_trainable(model)/1e3:.1f}K")

    trainer = Trainer(model, cfg, device)
    data_file = Path(cfg.data_dir) / "sft.jsonl"
    if not data_file.exists():
        sys.exit(f"缺少 {data_file}，请先运行 scripts/02_prepare_data.py --stage sft")
    ds = SftDataset(tok, data_file, cfg.max_seq_len, cfg.max_samples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft(b, tok.pad_id or 0),
    )

    def compute_loss(batch, model, device):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], labels=batch["labels"])
        return {"loss": out.loss}

    trainer.train_loop(loader, compute_loss)

    if cfg.merge_after:
        merge_lora(model)
        # 保存合并后的干净权重（不含 LoRA 结构键），可直接用于 DPO/导出
        merged = {k: v.cpu() for k, v in model.state_dict().items() if ".lora." not in k and "original_weight" not in k}
        out_path = Path(cfg.output_dir) / f"{cfg.run_name}_merged.pt"
        torch.save({"model": merged, "step": trainer.step, "config": cfg}, out_path)
        print(f"LoRA 已合并保存: {out_path}")


if __name__ == "__main__":
    main()
