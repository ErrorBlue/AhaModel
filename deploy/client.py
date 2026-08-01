#!/usr/bin/env python
"""独立 vLLM 对话客户端（OpenAI 兼容 API，仅依赖 requests）。"""

import argparse

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="ahamodel")
    parser.add_argument("--prompt", default=None, help="单次提问；不填则进入交互模式")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    url = f"{args.base_url}/chat/completions"
    history = []
    prompts = [args.prompt] if args.prompt else None

    def ask(prompt: str) -> str:
        history.append({"role": "user", "content": prompt})
        resp = requests.post(
            url,
            json={"model": args.model, "messages": history, "temperature": 0.8, "max_tokens": args.max_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": content})
        return content

    if prompts:
        print(ask(prompts[0]))
        return
    print(f"已连接 {url}，输入 exit 退出")
    while True:
        try:
            user = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit"):
            break
        print(f"模型> {ask(user)}")


if __name__ == "__main__":
    main()
