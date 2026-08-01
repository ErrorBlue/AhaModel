"""配置加载与 smoke 覆盖测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahamodel.config import (
    apply_smoke_model,
    apply_smoke_stage,
    load_model_config,
    load_stage_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_model_config_default():
    cfg = load_model_config(ROOT / "configs/model.yaml")
    assert cfg.d_model == 768 and cfg.n_layers == 8
    assert cfg.q_heads == 8 and cfg.kv_heads == 4
    assert cfg.head_dim == 96
    assert 50 < cfg.num_params_estimate < 80  # 64M 量级


def test_stage_config_and_smoke():
    cfg = load_stage_config(ROOT / "configs/pretrain.yaml")
    assert cfg.batch_size == 8 and cfg.max_seq_len == 768
    mcfg = apply_smoke_model(load_model_config(ROOT / "configs/model.yaml"))
    assert mcfg.d_model == 128 and mcfg.n_layers == 2
    scfg = apply_smoke_stage(cfg)
    assert scfg.max_steps == 6 and scfg.batch_size == 2


def test_unknown_fields_warn():
    import warnings

    import yaml

    bad = Path(__file__).resolve().parents[1] / "configs" / "_tmp_bad.yaml"
    bad.write_text("batch_size: 4\nnot_a_field: 123\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_stage_config(bad)
        assert any("未定义字段" in str(x.message) for x in w)
    bad.unlink()
