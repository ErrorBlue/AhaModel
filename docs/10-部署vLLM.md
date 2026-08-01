# 10 部署 vLLM（4090 服务器）

## 1. 导出 HF 格式

vLLM 需要标准 transformers 模型目录。本项目权重命名从第一天起就是 HF Llama 风格：

```bash
python scripts/11_export_hf.py --model-path checkpoints/sft/latest.pt
# 产物 checkpoints/hf/:
#   config.json（LlamaConfig）  model.safetensors  tokenizer.json  tokenizer_config.json
```

tokenizer 导出会把自写 BPE 转成标准 `tokenizer.json`（GPT-2 字节映射 + 特殊 token），
聊天模板内嵌在 `tokenizer_config.json`，vLLM 的 chat 端点可直接用。

## 2. 启动服务

```bash
# 方式一：直接跑（建议用 tmux 或 systemd 托管，见第 4 节）
bash deploy/vllm_serve.sh checkpoints/hf 8000

# 方式二：Docker（NVIDIA Container Toolkit 已装）
cd deploy && docker compose up
```

验证：

```bash
curl http://127.0.0.1:8000/v1/models
```

## 3. 客户端对话

```bash
python scripts/12_deploy_vllm.py --action client --base-url http://127.0.0.1:8000/v1
python deploy/client.py --prompt "你好，介绍一下你自己"
```

接口兼容 OpenAI Chat Completions，可接 FastGPT、Open-WebUI 等。

## 4. 长期运行（systemd）

```ini
# /etc/systemd/system/ahamodel-vllm.service
[Unit]
Description=AhaModel vLLM server
After=network.target

[Service]
WorkingDirectory=/home/user/AhaModel
ExecStart=/home/user/AhaModel/.venv/bin/vllm serve checkpoints/hf --served-model-name ahamodel --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 5. 常见问题

- **vLLM 不支持 Windows**：在 Linux/云 GPU 上运行（本项目的既定环境）；
- **max-model-len 过大**：显存不够时调小（64M 模型 1024 长度绰绰有余）；
- **导出后 logits 与训练不一致**：运行 `pytest tests/test_export.py -q`
  （transformers 加载后对比 logits，容差 1e-4）；
- **输出不遵循聊天格式**：确认 client 走 `/chat/completions`（应用 chat_template）。

## 扩展思路

- 量化（GPTQ/AWQ）与 GGUF/Ollama 部署；
- 用 vLLM 的 prefix caching 加速多轮对话；
- 基于本服务接 LangChain 做 Agent 实验。
