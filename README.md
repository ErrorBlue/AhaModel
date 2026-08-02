# AhaModel —— 从零构建 LLM 全流程教学项目

> 「读得懂、跑得动、改得开」：用 PyTorch 从零手写 Tokenizer、Transformer 与全阶段训练算法，
> 在单张 4090 上半天内跑通 预训练 → SFT → LoRA → DPO → PPO → GRPO → 评测 → vLLM 部署。

## 项目特点

- **全部核心算法从 0 实现**：字节级 BPE、Decoder-only Transformer（RMSNorm/RoPE/GQA/SwiGLU/KV Cache）、
  训练循环、LoRA、DPO、RM、PPO（GAE）、GRPO（组内 advantage），不依赖 peft/trl 等高层封装；
- **生态只用在刀刃上**：`huggingface_hub/datasets` 下载数据、`transformers/safetensors` 导出与评测、`vLLM` 部署；
- **4090 半天跑完**：默认 64M 模型 + minimind 开源 mini 数据集，约 1~2h 预训练、1~2h SFT、其余各 0.5~1h；
- **中文教学文档**：每个阶段一篇 docs/ 讲解（原理 + 公式 + 代码导读 + 运行命令）；
- **工程化齐全**：`--smoke` 快速验证、断点续训、SwanLab/wandb 监控、HF 格式导出、Docker/vLLM 部署脚本。

## 学习路线

| 步骤 | 脚本 | 文档 | 产物 |
| --- | --- | --- | --- |
| 环境搭建 | `deploy/server_setup.sh` | [docs/00](docs/00-环境搭建.md) | 服务器环境 |
| 数据准备 | `scripts/01_prepare_data.py` | [docs/01](docs/01-数据准备与清洗.md) | `data/{pretrain,sft,dpo,rlaif}.jsonl` |
| Tokenizer | `scripts/02_train_tokenizer.py` | [docs/02](docs/02-Tokenizer.md) | `data/tokenizer.json` |
| 预训练 | `scripts/03_pretrain.py` | [docs/04](docs/04-预训练.md) | `checkpoints/pretrain/` |
| SFT | `scripts/04_sft.py` | [docs/05](docs/05-SFT与LoRA.md) | `checkpoints/sft/` |
| LoRA 微调 | `scripts/05_lora.py` | [docs/05](docs/05-SFT与LoRA.md) | `checkpoints/lora_merged.pt` |
| DPO | `scripts/06_dpo.py` | [docs/06](docs/06-RLHF-DPO.md) | `checkpoints/dpo/` |
| 奖励模型 | `scripts/07_rm.py` | [docs/07](docs/07-RLHF-PPO.md) | `checkpoints/rm/` |
| PPO | `scripts/08_ppo.py` | [docs/07](docs/07-RLHF-PPO.md) | `checkpoints/ppo/` |
| GRPO | `scripts/09_grpo.py` | [docs/08](docs/08-RLHF-GRPO.md) | `checkpoints/grpo/` |
| 评测 | `scripts/10_eval.py` | [docs/09](docs/09-评测.md) | `results/` |
| 导出 HF | `scripts/11_export_hf.py` | [docs/10](docs/10-部署vLLM.md) | `checkpoints/hf/` |
| vLLM 部署 | `scripts/12_deploy_vllm.py` | [docs/10](docs/10-部署vLLM.md) | OpenAI API 服务 |

## 快速开始

数据准备与 tokenizer 两条路径共用，先准备一次（需要联网下载数据）：

```bash
# 0. 安装项目（两条路径都需执行一次）
pip install -e .

# 1. 下载并清洗数据（国内服务器：export HF_ENDPOINT=https://hf-mirror.com）
python scripts/01_prepare_data.py --stage all

# 2. 基于清洗后的语料训练 tokenizer（真实语料建议 --sample 200000）
python scripts/02_train_tokenizer.py
```

之后按目标二选一。

### 方案 A：CPU smoke —— 几分钟验证全链路

`--smoke` 把模型缩到 2 层 / 128 维、数据限制 200 条、每阶段只跑 6 步，配合 `--device cpu`
可在无 GPU 的机器上几分钟跑通 预训练 → SFT → LoRA → DPO → RM → PPO → GRPO → 评测 → 导出。
vLLM 部署仅支持 GPU/Linux，本路径止步于 HF 导出（产物可用 transformers 加载验证）。

```bash
# 预训练 -> SFT -> LoRA
python scripts/03_pretrain.py --smoke --device cpu
python scripts/04_sft.py --smoke --device cpu --model-path checkpoints/pretrain/latest.pt
python scripts/05_lora.py --smoke --device cpu --model-path checkpoints/sft/latest.pt

# 对齐与 RL
python scripts/06_dpo.py --smoke --device cpu --model-path checkpoints/sft/latest.pt
python scripts/07_rm.py --smoke --device cpu --model-path checkpoints/sft/latest.pt
python scripts/08_ppo.py --smoke --device cpu --model-path checkpoints/sft/latest.pt --rm-path checkpoints/rm/latest.pt
python scripts/09_grpo.py --smoke --device cpu --model-path checkpoints/sft/latest.pt

# 评测与导出（--smoke/--device 需放在 eval 子命令 samples 之前）
python scripts/10_eval.py --smoke --device cpu samples --model-path checkpoints/sft/latest.pt --chat
python scripts/11_export_hf.py --model-path checkpoints/sft/latest.pt
```

### 方案 B：GPU 正式训练（4090 服务器，半天跑完）

默认 64M 模型 + minimind 数据集，约 1~2h 预训练、1~2h SFT、其余各 0.5~1h。

```bash
# 环境（Ubuntu + NVIDIA 驱动）
bash deploy/server_setup.sh

# 预训练 -> SFT -> LoRA
python scripts/03_pretrain.py
python scripts/04_sft.py --model-path checkpoints/pretrain/latest.pt
python scripts/05_lora.py --model-path checkpoints/sft/latest.pt

# 对齐与 RL
python scripts/06_dpo.py --model-path checkpoints/sft/latest.pt
python scripts/07_rm.py --model-path checkpoints/sft/latest.pt
python scripts/08_ppo.py --model-path checkpoints/sft/latest.pt --rm-path checkpoints/rm/latest.pt
python scripts/09_grpo.py --model-path checkpoints/sft/latest.pt

# 评测与部署
python scripts/10_eval.py samples --model-path checkpoints/sft/latest.pt --chat
python scripts/11_export_hf.py --model-path checkpoints/sft/latest.pt
bash deploy/vllm_serve.sh
python scripts/12_deploy_vllm.py --action client
```

注意：模型规模由 checkpoint 决定，两条路径不要混用——用 smoke 预训练产物接正式 SFT
得到的仍是小模型；正式训练需从 `python scripts/03_pretrain.py`（不带 `--smoke`）开始。

## 项目结构

```
AhaModel/
├── README.md / docs/          # 教学文档（00-10，与脚本编号对应）
├── ahamodel/                  # 核心包
│   ├── config.py              # 统一配置（dataclass + yaml）
│   ├── tokenizer/             # 手写字节级 BPE
│   ├── model/                 # Transformer（RMSNorm/RoPE/GQA/SwiGLU/KV Cache）+ 生成
│   ├── data/                  # 下载、清洗、数据集、聊天模板、math 规则奖励
│   ├── train/                 # trainer、LoRA、DPO、RM、PPO、GRPO
│   ├── eval/                  # PPL、生成样例、模型对比、lm-eval 接入
│   ├── export/                # HF 格式导出（config.json + safetensors + tokenizer.json）
│   └── utils/                 # 命令行公共参数
├── scripts/                   # 编号入口 01-12
├── configs/                   # model.yaml + 各阶段 yaml
├── deploy/                    # server_setup.sh / vllm_serve.sh / docker-compose.yml / client.py
├── tests/                     # pytest（25 项单测）
├── data/ checkpoints/ results/   # 训练产物（gitignore）
└── requirements.txt / pyproject.toml
```

## 配置与规模

`configs/model.yaml` 默认 64M（`d_model=768, n_layers=8, q_heads=8, kv_heads=4, vocab=6400`），
另有 26M 快速档与 150M 增强档（150M 训练时长约为 64M 的 2~3 倍）。所有训练超参集中在
`configs/*.yaml`，命令行可用 `--lr/--batch-size/--max-steps/--max-samples/--use-logger` 覆盖。

## 测试

```bash
pip install pytest
pytest tests/ -q
```

单测覆盖：BPE 编解码往返与朴素实现对照、模型前向/反向、KV Cache 一致性、padding 无 NaN、
LoRA 合并/解合并等价、聊天模板掩码、GAE 手算、DPO 损失方向、trainer 断点续训、HF 导出一致性。

## 留作扩展（TODO）

MoE、模型蒸馏、量化与 GGUF/Ollama、Agent/Tool 训练、DDP/FSDP 多卡、YaRN 长文本外推、
WebUI。每个方向在对应 docs 末尾都有「扩展思路」小节。
