"""可选第三方基准：lm-evaluation-harness（C-Eval 等）。"""

from __future__ import annotations

from pathlib import Path


def run_lm_eval(export_dir: str | Path, tasks: str = "ceval-valid", limit: int | None = None) -> None:
    """包装 lm-eval-harness。需要已导出 HF 格式模型（scripts/11_export_hf.py）。

    示例：
      pip install lm_eval
      lm_eval --model hf --model_args pretrained=checkpoints/hf \
              --tasks ceval-valid --batch_size 8 [--limit 100]
    """
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print(
            "lm_eval 未安装。请先: pip install lm_eval\n"
            f"然后运行:\n  lm_eval --model hf --model_args pretrained={export_dir} "
            f"--tasks {tasks}" + (f" --limit {limit}" if limit else "")
        )
        return
    from lm_eval import simple_evaluate

    args = {
        "model": "hf",
        "model_args": f"pretrained={export_dir}",
        "tasks": tasks,
        "batch_size": 8,
        "device": "cuda",
    }
    if limit:
        args["limit"] = limit
    results = simple_evaluate(**args)
    print(results.get("results", {}))
