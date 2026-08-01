"""把 AhaModel 权重导出为 HF Llama 格式：config.json + safetensors + tokenizer。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ahamodel.config import ModelConfig
from ahamodel.export.tokenizer_hf import export_hf_tokenizer


def export_to_hf(model, tokenizer, cfg: ModelConfig, out_dir: str | Path) -> Path:
    """导出目录可直接被 transformers / vLLM / lm-eval 加载。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.q_heads,
        "num_key_value_heads": cfg.kv_heads,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "hidden_act": "silu",
        "attention_dropout": cfg.dropout,
        "initializer_range": 0.02,
        "tie_word_embeddings": cfg.tie_embeddings,
        "use_cache": True,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "pad_token_id": tokenizer.pad_id,
        "torch_dtype": "float32",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 权重命名从第一天起就是 HF Llama 风格，直接转存即可
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().to(torch.float32).contiguous() for k, v in model.state_dict().items()}
    save_file(state, out_dir / "model.safetensors")

    export_hf_tokenizer(tokenizer, out_dir, model_max_length=cfg.max_seq_len)

    gen_config = {
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "pad_token_id": tokenizer.pad_id,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_new_tokens": 256,
    }
    with open(out_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_config, f, ensure_ascii=False, indent=2)

    print(f"HF 格式模型已导出到 {out_dir}")
    print("本地验证: python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
          "m = AutoModelForCausalLM.from_pretrained(r'%s'); print(m.num_parameters())\" " % out_dir)
    return out_dir
