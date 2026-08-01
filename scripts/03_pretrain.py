"""步骤 3：预训练（next-token prediction）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from ahamodel.data.dataset import PretrainDataset, collate_lm
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.trainer import Trainer
from ahamodel.utils.cli import add_common_args, resume_trainer, setup_stage


def main():
    parser = argparse.ArgumentParser(description="预训练")
    add_common_args(parser)
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/pretrain.yaml", ROOT)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size  # 以 tokenizer 实际词表为准
    model = AhaForCausalLM(mcfg).to(device)
    print(f"参数量: {model.num_params()/1e6:.2f}M")

    trainer = Trainer(model, cfg, device)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    if args.from_resume:
        resume_trainer(trainer, run_dir, device)

    data_file = Path(cfg.data_dir) / "pretrain.jsonl"
    if not data_file.exists():
        sys.exit(f"缺少 {data_file}，请先运行 scripts/02_prepare_data.py --stage pretrain")
    ds = PretrainDataset(tok, data_file, cfg.max_seq_len, cfg.max_samples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_lm(b, tok.pad_id or 0),
    )

    def compute_loss(batch, model, device):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], labels=batch["labels"])
        return {"loss": out.loss}

    trainer.train_loop(loader, compute_loss)


if __name__ == "__main__":
    main()
