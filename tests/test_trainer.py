"""Trainer 冒烟：真实小数据上跑几步 + checkpoint 存取。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("torch")
import torch
from torch.utils.data import DataLoader

from ahamodel.config import ModelConfig, StageConfig
from ahamodel.data.dataset import PretrainDataset, collate_lm
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import BPE, SPECIAL_TOKENS, Tokenizer
from ahamodel.train.trainer import Trainer


def test_trainer_smoke_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    data = tmp_path / "pretrain.jsonl"
    texts = ["今天天气不错，我们出去玩。" * 3, "机器学习是人工智能的重要方向。" * 3] * 10
    with open(data, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    tok = Tokenizer(
        BPE(SPECIAL_TOKENS, max_vocab_size=300).train(["今天天气不错，我们出去玩。", "机器学习是人工智能的重要方向。"], min_freq=1, verbose=False)
    )
    model = AhaForCausalLM(
        ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, q_heads=4, kv_heads=2, intermediate_size=128, max_seq_len=64)
    )
    cfg = StageConfig(
        data_dir=str(tmp_path), output_dir=str(tmp_path / "ckpt"), run_name="smoke",
        batch_size=2, grad_accum=2, max_steps=4, log_every=1, save_every=2,
        max_seq_len=64, use_logger="none", lr=1e-3,
    )
    trainer = Trainer(model, cfg, "cpu")
    ds = PretrainDataset(tok, data, 64)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_lm(b, tok.pad_id or 0))

    def compute_loss(batch, model, device):
        batch = {k: v.to(device) for k, v in batch.items()}
        return {"loss": model(batch["input_ids"], labels=batch["labels"]).loss}

    trainer.train_loop(loader, compute_loss)
    assert trainer.step == 4
    latest = Path(cfg.output_dir) / cfg.run_name / "latest.pt"
    assert latest.exists()
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 4

    # 断点续训：新 trainer 恢复后继续
    model2 = AhaForCausalLM(
        ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, q_heads=4, kv_heads=2, intermediate_size=128, max_seq_len=64)
    )
    trainer2 = Trainer(model2, cfg, "cpu")
    trainer2.load_state_dict(ckpt)
    assert trainer2.step == 4
