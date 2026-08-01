"""步骤 6：DPO 直接偏好优化（从零实现）。"""

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ahamodel.data.dataset import DpoDataset, collate_dpo
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.rl import response_logprobs
from ahamodel.train.trainer import Trainer
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, setup_stage


def main():
    parser = argparse.ArgumentParser(description="DPO")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="SFT checkpoint（策略初始权重）")
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/dpo.yaml", ROOT)
    model_path = args.model_path or cfg.model_path
    mcfg = infer_model_config(model_path, mcfg, device)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size
    model = AhaForCausalLM(mcfg)
    load_pretrained(model, model_path, device)
    model = model.to(device)

    reference = copy.deepcopy(model).to(device).eval()
    for p in reference.parameters():
        p.requires_grad = False

    trainer = Trainer(model, cfg, device)
    data_file = Path(cfg.data_dir) / "dpo.jsonl"
    if not data_file.exists():
        sys.exit(f"缺少 {data_file}，请先运行 scripts/02_prepare_data.py --stage dpo")
    ds = DpoDataset(tok, data_file, cfg.max_seq_len, cfg.max_samples)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: collate_dpo(b, tok.pad_id or 0))

    def compute_loss(batch, model, device):
        ch = {k: v.to(device) for k, v in batch["chosen"].items()}
        rj = {k: v.to(device) for k, v in batch["rejected"].items()}
        with torch.no_grad():
            ref_ch, _ = response_logprobs(reference, ch["input_ids"], ch["labels"], device)
            ref_rj, _ = response_logprobs(reference, rj["input_ids"], rj["labels"], device)
        logp_ch, _ = response_logprobs(model, ch["input_ids"], ch["labels"], device)
        logp_rj, _ = response_logprobs(model, rj["input_ids"], rj["labels"], device)
        # DPO 损失：-log σ(β * [(π(yw)-π_ref(yw)) - (π(yl)-π_ref(yl))])
        inner = (logp_ch - logp_rj) - (ref_ch - ref_rj)
        loss = -F.logsigmoid(cfg.beta * inner).mean()
        acc = ((logp_ch - ref_ch) > (logp_rj - ref_rj)).float().mean()
        return {"loss": loss, "dpo_acc": acc}

    trainer.train_loop(loader, compute_loss)


if __name__ == "__main__":
    main()
