"""步骤 4：指令微调（全参数），loss 只算 assistant 回答。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from ahamodel.data.dataset import SftDataset, collate_sft
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.trainer import Trainer
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, resume_trainer, setup_stage


def main():
    parser = argparse.ArgumentParser(description="SFT")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="预训练 checkpoint（.pt）")
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/sft.yaml", ROOT)
    model_path = args.model_path or cfg.model_path
    mcfg = infer_model_config(model_path, mcfg, device)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size
    model = AhaForCausalLM(mcfg)
    if model_path:
        load_pretrained(model, model_path, device)
    model = model.to(device)
    print(f"参数量: {model.num_params()/1e6:.2f}M")

    trainer = Trainer(model, cfg, device)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    if args.from_resume:
        resume_trainer(trainer, run_dir, device)

    data_file = Path(cfg.data_dir) / "sft.jsonl"
    if not data_file.exists():
        sys.exit(f"缺少 {data_file}，请先运行 scripts/01_prepare_data.py --stage sft")
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


if __name__ == "__main__":
    main()
