"""步骤 12：vLLM 部署（打印命令 / 交互式 client）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def print_serve_commands(model_dir: str, port: int) -> None:
    print("=" * 60)
    print("vLLM 服务启动（Linux 服务器）:")
    print(f"  vllm serve {model_dir} --served-model-name ahamodel --port {port} --max-model-len 1024")
    print("或在项目根目录运行:")
    print(f"  bash deploy/vllm_serve.sh {model_dir} {port}")
    print("Docker 方式:")
    print("  cd deploy && docker compose up")
    print("=" * 60)


def run_client(base_url: str, model: str) -> None:
    import requests

    url = f"{base_url}/chat/completions"
    print(f"连接到 {url}，输入 exit 退出")
    history = []
    while True:
        user = input("用户> ").strip()
        if user in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user})
        resp = requests.post(
            url,
            json={"model": model, "messages": history, "temperature": 0.8, "max_tokens": 256},
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": content})
        print(f"模型> {content}")


def main():
    parser = argparse.ArgumentParser(description="vLLM 部署辅助")
    parser.add_argument("--action", choices=["serve", "client", "info"], default="info")
    parser.add_argument("--model-dir", default="checkpoints/hf")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="ahamodel")
    args = parser.parse_args()

    if args.action == "serve":
        print_serve_commands(args.model_dir, args.port)
    elif args.action == "client":
        run_client(args.base_url, args.model)
    else:
        print_serve_commands(args.model_dir, args.port)
        print("\n启动服务后可运行:")
        print(f"  python scripts/12_deploy_vllm.py --action client --base-url {args.base_url}")


if __name__ == "__main__":
    main()
