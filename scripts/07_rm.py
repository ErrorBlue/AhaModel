"""步骤 7：奖励模型训练（pair 排序损失，PPO 的奖励信号来源）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ahamodel.data.dataset import DpoDataset, collate_dpo
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.reward_model import RewardModel
from ahamodel.train.trainer import Trainer
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, setup_stage


def main():
    parser = argparse.ArgumentParser(description="奖励模型训练")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="SFT checkpoint 初始化骨干（可选）")
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/rm.yaml", ROOT)
    model_path = args.model_path or cfg.model_path
    mcfg = infer_model_config(model_path, mcfg, device)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size
    model = RewardModel(mcfg)
    if model_path:
        load_pretrained(model, model_path, device, strict=False)
    model = model.to(device)

    trainer = Trainer(model, cfg, device)
    data_file = Path(cfg.data_dir) / "dpo.jsonl"
    if not data_file.exists():
        sys.exit(f"缺少 {data_file}，请先运行 scripts/01_prepare_data.py --stage dpo")
    ds = DpoDataset(tok, data_file, cfg.max_seq_len, cfg.max_samples)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: collate_dpo(b, tok.pad_id or 0))

    def compute_loss(batch, model, device):
        ch = {k: v.to(device) for k, v in batch["chosen"].items()}
        rj = {k: v.to(device) for k, v in batch["rejected"].items()}
        r_ch = model(ch["input_ids"], ch["response_start"], ch["attention_mask"])
        r_rj = model(rj["input_ids"], rj["response_start"], rj["attention_mask"])
        loss = -F.logsigmoid(r_ch - r_rj - cfg.rm_margin).mean()
        acc = (r_ch > r_rj).float().mean()
        return {"loss": loss, "rm_acc": acc}

    trainer.train_loop(loader, compute_loss)


if __name__ == "__main__":
    main()
