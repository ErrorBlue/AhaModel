"""生成样例对比。"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ahamodel.model.generate import generate

DEFAULT_PROMPTS = [
    "你好，请介绍一下你自己。",
    "中国的首都是哪里？",
    "请用一句话解释什么是机器学习。",
    "7乘以8等于多少？",
    "写一段关于秋天的短文。",
]


def generate_samples(
    model,
    tokenizer,
    prompts: Optional[Sequence[str]] = None,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cpu",
    chat: bool = False,
    use_cache: bool = True,
) -> List[str]:
    from ahamodel.data.template import ChatTemplate

    prompts = list(prompts or DEFAULT_PROMPTS)
    template = ChatTemplate(tokenizer)
    outputs = []
    for prompt in prompts:
        if chat:
            # 停在 <|assistant|> 标记处，让模型续写回答（而不是在 eos 之后生成）
            prompt_ids = template.encode_rollout_prompt(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
            )
        else:
            prompt_ids = tokenizer.encode(prompt)
        gen = generate(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
            use_cache=use_cache,
        )
        text = tokenizer.decode(gen)
        if chat:
            # 去掉开头的 <|user|>... 部分，只展示 assistant 回复
            idx = text.rfind("<|assistant|>")
            shown = text[idx + len("<|assistant|>"):] if idx != -1 else text
            shown = shown.replace("<|eos|>", "").strip()
        else:
            shown = text
        outputs.append(shown)
        print(f"用户: {prompt}\n模型: {shown}\n{'=' * 50}")
    return outputs
