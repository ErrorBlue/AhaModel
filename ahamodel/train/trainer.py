"""通用训练循环：AMP、梯度累积、余弦退火、断点续训、可视化。"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ahamodel.config import StageConfig


class Logger:
    """极简日志封装：console / tensorboard / swanlab / wandb。"""

    def __init__(self, backend: str = "none", run_name: str = "run", output_dir: str = "checkpoints"):
        self.backend = backend
        self.run_name = run_name
        self.writer = None
        if backend == "none":
            return
        try:
            if backend == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                log_dir = str(Path(output_dir) / run_name / "tensorboard")
                self.writer = SummaryWriter(log_dir=log_dir)
            elif backend == "swanlab":
                import swanlab

                swanlab.init(project="AhaModel", name=run_name)
                self.writer = swanlab
            elif backend == "wandb":
                import wandb

                wandb.init(project="AhaModel", name=run_name)
                self.writer = wandb
        except Exception as e:  # 可视化库缺失/登录失败时降级为 console
            warnings.warn(f"Logger({backend}) 初始化失败，降级为 console: {e}")
            self.backend = "none"

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.backend == "tensorboard" and self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        elif self.backend in ("swanlab", "wandb") and self.writer is not None:
            self.writer.log(metrics, step=step)

    def finish(self) -> None:
        if self.writer is not None:
            try:
                if self.backend == "tensorboard":
                    self.writer.close()
                elif self.backend in ("swanlab", "wandb"):
                    self.writer.finish()
            except Exception:
                pass


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """warmup 线性上升 + 余弦退火。"""
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg: StageConfig, device: str, logger: Optional[Logger] = None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.logger = logger or Logger(cfg.use_logger, cfg.run_name, cfg.output_dir)
        self.step = 0

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("模型没有可训练参数（请检查 LoRA 是否应用成功）")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=tuple(cfg.betas)
        )

        # AMP 策略：GPU 上 bf16 优先，无 bf16 支持时回退 fp16，CPU 一律 fp32
        self.amp_dtype = None
        self.scaler = None
        if device.startswith("cuda"):
            if cfg.dtype == "bf16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            elif cfg.dtype in ("bf16", "fp16"):
                self.amp_dtype = torch.float16
                try:
                    self.scaler = torch.amp.GradScaler("cuda", enabled=True)
                except TypeError:
                    self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.total_steps = cfg.max_steps
        self.trainable_params = trainable

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------
    def state_dict(self, extra: Optional[dict] = None) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "config": self.cfg,
            "model_config": getattr(self.model, "cfg", None),  # 模型结构配置，加载时自动恢复
            "extra": extra or {},
        }

    def load_state_dict(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt.get("step", 0)

    def save_checkpoint(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(extra), path)

    def latest_checkpoint(self, run_dir: str | Path) -> Optional[Path]:
        p = Path(run_dir) / "latest.pt"
        return p if p.exists() else None

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def train_loop(
        self,
        dataloader: DataLoader,
        compute_loss: Callable[[dict, torch.nn.Module, str], Dict[str, torch.Tensor]],
        total_steps: Optional[int] = None,
        eval_fn: Optional[Callable[[torch.nn.Module, int], dict]] = None,
    ) -> None:
        total_steps = total_steps or self.total_steps
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        accum = max(1, cfg.grad_accum)
        run_dir = Path(cfg.output_dir) / cfg.run_name
        start_time = time.time()
        micro = 0
        tokens_seen = 0

        model.train()
        optimizer.zero_grad()
        while self.step < total_steps:
            for batch in dataloader:
                if self.step >= total_steps:
                    break
                out = compute_loss(batch, model, self.device)
                loss = out["loss"] / accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro += 1
                tokens_seen += int(batch.get("input_ids", torch.zeros(1)).numel())

                if micro % accum == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, cfg.grad_clip)
                    if self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    self.step += 1

                    # 手动余弦退火（教学：不引入第三方 scheduler）
                    lr_now = cosine_lr(self.step, total_steps, cfg.warmup_steps, cfg.lr)
                    for g in optimizer.param_groups:
                        g["lr"] = lr_now

                    if self.step % cfg.log_every == 0:
                        speed = tokens_seen / max(1e-6, time.time() - start_time)
                        metrics = {
                            "loss": float(out["loss"].detach().item()),
                            "lr": lr_now,
                            "tokens_per_sec": speed,
                            "step": self.step,
                        }
                        print(
                            f"[{cfg.run_name}] step {self.step}/{total_steps} "
                            f"loss {metrics['loss']:.4f} lr {lr_now:.2e} {speed:.0f} tok/s"
                        )
                        self.logger.log(metrics, self.step)
                        tokens_seen = 0
                        start_time = time.time()

                    if cfg.save_every > 0 and self.step % cfg.save_every == 0:
                        self.save_checkpoint(run_dir / f"step_{self.step}.pt")
                        self.save_checkpoint(run_dir / "latest.pt")

                    if cfg.eval_every > 0 and self.step % cfg.eval_every == 0 and eval_fn is not None:
                        model.eval()
                        metrics = eval_fn(model, self.step)
                        model.train()
                        self.logger.log(metrics, self.step)

        self.save_checkpoint(run_dir / "latest.pt")
        self.logger.finish()
        print(f"训练完成，共 {self.step} 步，checkpoint 保存在 {run_dir}")
