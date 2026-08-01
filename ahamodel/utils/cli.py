"""命令行公共参数与初始化辅助。"""

from __future__ import annotations

import argparse
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from ahamodel.config import (
    ModelConfig,
    StageConfig,
    apply_smoke_model,
    apply_smoke_stage,
    load_model_config,
    load_stage_config,
    resolve_device,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="阶段 yaml 配置路径")
    parser.add_argument("--model-config", default=None, help="模型结构 yaml 配置路径")
    parser.add_argument("--smoke", action="store_true", help="微型模式：CPU 上几分钟跑通")
    parser.add_argument("--from-resume", action="store_true", help="从 checkpoints/<run>/latest.pt 恢复")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-logger", default=None)


def setup_stage(args: argparse.Namespace, default_config: str, repo_root: Path):
    """解析阶段配置 + 模型配置，应用 --smoke/覆盖项，返回 (stage_cfg, model_cfg, device)。"""
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k in {f.name for f in fields(StageConfig)}
    }
    cfg = load_stage_config(args.config or str(repo_root / default_config), **overrides)
    model_cfg = load_model_config(args.model_config or str(repo_root / "configs/model.yaml"))
    if args.smoke:
        model_cfg = apply_smoke_model(model_cfg)
        cfg = apply_smoke_stage(cfg)
    device = resolve_device(args.device or cfg.device)
    seed_everything(cfg.seed)
    print(f"device={device} run={cfg.run_name} dtype={cfg.dtype} "
          f"batch={cfg.batch_size} grad_accum={cfg.grad_accum} max_steps={cfg.max_steps}")
    return cfg, model_cfg, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained(model: torch.nn.Module, path: str | Path, device: str, strict: bool = True) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    print(f"已加载权重: {path}")


def infer_model_config(model_path, fallback: ModelConfig, device: str = "cpu") -> ModelConfig:
    """优先从 checkpoint 读取模型结构配置（如 --smoke 产物），否则用命令行配置。"""
    if not model_path:
        return fallback
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        mc = ckpt.get("model_config")
        if isinstance(mc, ModelConfig):
            print(f"从 checkpoint 恢复模型结构: {mc.name} (d_model={mc.d_model}, layers={mc.n_layers})")
            return mc
    except Exception:
        pass
    return fallback


def resume_trainer(trainer, run_dir: str | Path, device: str) -> bool:
    latest = Path(run_dir) / "latest.pt"
    if not latest.exists():
        return False
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    trainer.load_state_dict(ckpt)
    print(f"断点续训: 恢复至 step {trainer.step}")
    return True
