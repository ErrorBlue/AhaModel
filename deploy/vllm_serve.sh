#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务
# 用法: bash deploy/vllm_serve.sh [模型目录] [端口]
set -euo pipefail

MODEL_DIR="${1:-checkpoints/hf}"
PORT="${2:-8000}"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "找不到 $MODEL_DIR/config.json，请先运行: python scripts/11_export_hf.py --model-path checkpoints/<run>/latest.pt"
  exit 1
fi

vllm serve "$MODEL_DIR" \
  --served-model-name ahamodel \
  --port "$PORT" \
  --max-model-len 1024 \
  --enforce-eager
