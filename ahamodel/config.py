"""统一配置：dataclass 定义 + yaml 加载/保存 + smoke 覆盖。

设计原则：所有超参数集中在这里，方便通读与修改；脚本只负责
解析命令行参数（--config / --model-config / --smoke 等）并传入。
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def dataclass_from(cls, data: dict | None) -> Any:
    """从 dict 构造 dataclass：只取已知字段，未知键给出警告（便于教学调试）。"""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        warnings.warn(f"{cls.__name__} 收到未定义字段，已忽略: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def dataclass_to_dict(obj: Any) -> dict:
    return asdict(obj)


@dataclass
class ModelConfig:
    """Transformer 模型结构配置。"""

    name: str = "ahamodel-64m"
    vocab_size: int = 6400
    d_model: int = 768
    n_layers: int = 8
    q_heads: int = 8
    kv_heads: int = 4
    intermediate_size: int = 2048
    max_seq_len: int = 768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = False
    # YaRN 等 RoPE 外推方案（v1 留作扩展）
    rope_scaling: Optional[dict] = None

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.q_heads == 0, "d_model 必须能被 q_heads 整除"
        return self.d_model // self.q_heads

    @property
    def num_params_estimate(self) -> float:
        """粗略参数量估算（MB 级），用于教学演示。"""
        n = 2 * self.vocab_size * self.d_model if not self.tie_embeddings else self.vocab_size * self.d_model
        per_layer = 3 * self.d_model * self.d_model  # q/k/v
        per_layer += self.d_model * self.d_model  # o
        per_layer += 3 * self.d_model * self.intermediate_size  # swiglu
        n += self.n_layers * per_layer
        return n / 1e6


@dataclass
class StageConfig:
    """训练/评测阶段的通用配置，含各阶段扩展字段（按需使用）。"""

    # ---- 通用 ----
    data_dir: str = "data"
    output_dir: str = "checkpoints"
    tokenizer_path: str = "data/tokenizer.json"
    model_path: Optional[str] = None
    seed: int = 42
    device: str = "auto"
    dtype: str = "bf16"
    run_name: str = "run"
    use_logger: str = "none"
    max_samples: Optional[int] = None

    # ---- 训练 ----
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 100
    max_steps: int = 5000
    epochs: int = 1
    max_seq_len: int = 512
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 0

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list = None
    merge_after: bool = True

    # ---- DPO / RM ----
    beta: float = 0.1
    ref_model_path: Optional[str] = None
    rm_margin: float = 0.0
    rm_path: Optional[str] = None

    # ---- PPO / GRPO ----
    response_max_len: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    kl_coef: float = 0.1
    ppo_epochs: int = 1
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    group_size: int = 8
    n_rollout_prompts: int = 1000
    reward_type: str = "math"

    # ---- 评测 ----
    eval_mode: str = "ppl"
    ppl_file: str = "data/pretrain_eval.jsonl"
    prompt_file: Optional[str] = None
    max_new_tokens: int = 96
    test_n: int = 100
    compare_sft_path: Optional[str] = None
    compare_rl_path: Optional[str] = None

    # ---- 导出 ----
    export_dir: str = "checkpoints/hf"

    def __post_init__(self):
        if self.lora_targets is None:
            self.lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_config(path: str | Path | None = None, **overrides) -> ModelConfig:
    data = load_yaml(path) if path else {}
    cfg = dataclass_from(ModelConfig, data)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_stage_config(path: str | Path | None = None, **overrides) -> StageConfig:
    data = load_yaml(path) if path else {}
    cfg = dataclass_from(StageConfig, data)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def apply_smoke_model(cfg: ModelConfig) -> ModelConfig:
    """--smoke：缩到可在 CPU 上 2 分钟跑通的规模。"""
    cfg.d_model = 128
    cfg.n_layers = 2
    cfg.q_heads = 4
    cfg.kv_heads = 2
    cfg.intermediate_size = 512
    cfg.max_seq_len = 128
    cfg.name = f"{cfg.name}-smoke"
    return cfg


def apply_smoke_stage(cfg: StageConfig) -> StageConfig:
    """--smoke：压缩训练规模与数据量。"""
    cfg.batch_size = 2
    cfg.grad_accum = 1
    cfg.max_steps = 6
    cfg.max_samples = 200
    cfg.max_seq_len = min(cfg.max_seq_len, 128)
    cfg.response_max_len = min(cfg.response_max_len, 24)
    cfg.max_new_tokens = min(cfg.max_new_tokens, 24)
    cfg.log_every = 1
    cfg.save_every = 10**9
    cfg.warmup_steps = 1
    cfg.use_logger = "none"
    cfg.lr = 1e-3
    return cfg


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
