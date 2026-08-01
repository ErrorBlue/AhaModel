"""GRPO 规则奖励用的小学算术数据：无需外部数据，开箱即用。"""

from __future__ import annotations

import random
import re
from typing import Optional


def make_math_prompts(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*"])
        a = rng.randint(1, 999)
        b = rng.randint(1, 999)
        if op == "-":
            a, b = max(a, b), min(a, b)
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        items.append({
            "prompt": f"请计算：{a} {op} {b} = ? 请只输出答案数字。",
            "answer": str(ans),
        })
    return items


def extract_answer(response: str) -> Optional[str]:
    """提取回复中最后一个整数（负数也支持）。"""
    matches = re.findall(r"-?\d+", response)
    return matches[-1] if matches else None


def rule_reward(item: dict, response: str) -> float:
    """规则奖励：答案完全匹配得 1 分，否则 0 分。

    教学提示：真实 RLVR 场景会把规则奖励换成单元测试、代码执行、
    格式校验等“可验证信号”；这里用最简单形式演示同一套接口。
    """
    ans = extract_answer(response)
    if ans is None:
        return 0.0
    return 1.0 if ans == str(item["answer"]) else 0.0
