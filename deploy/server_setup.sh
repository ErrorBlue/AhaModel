#!/usr/bin/env bash
# 4090 服务器环境初始化（Ubuntu 22.04+，NVIDIA 驱动与基础工具已装好）
# 用法: bash deploy/server_setup.sh
set -euo pipefail

echo "==> 安装 uv（Python 环境管理）"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> 创建 Python 3.12 虚拟环境"
uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate

echo "==> 安装 PyTorch（CUDA 12.8）与项目依赖"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

echo "==> 安装 vLLM（推理部署）"
pip install vllm

echo "==> 验证"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cat <<'EOF'
完成。下一步：
  1) 数据下载（国内服务器如无法直连 HF）:
     export HF_ENDPOINT=https://hf-mirror.com
     python scripts/01_prepare_data.py --stage all
  2) 训练:
     python scripts/03_pretrain.py --use-logger swanlab
EOF
