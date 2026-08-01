"""步骤 10：评测（PPL / 生成样例 / RL 对比 / lm-eval）。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahamodel.data.dataset import read_jsonl
from ahamodel.eval.compare import compare_models
from ahamodel.eval.generate_samples import DEFAULT_PROMPTS, generate_samples
from ahamodel.eval.perplexity import evaluate_ppl
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.reward_model import RewardModel
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, setup_stage


def main():
    parser = argparse.ArgumentParser(description="评测")
    add_common_args(parser)
    sub = parser.add_subparsers(dest="eval_mode", required=True)

    p1 = sub.add_parser("ppl", help="困惑度")
    p1.add_argument("--model-path", required=True)
    p1.add_argument("--ppl-file", default="data/pretrain_eval.jsonl")

    p2 = sub.add_parser("samples", help="生成样例")
    p2.add_argument("--model-path", required=True)
    p2.add_argument("--chat", action="store_true")
    p2.add_argument("--prompt-file", default=None)
    p2.add_argument("--max-new-tokens", type=int, default=None)

    p3 = sub.add_parser("compare", help="RL 模型 vs SFT 模型（RM 打分）")
    p3.add_argument("--sft-path", required=True)
    p3.add_argument("--rl-path", required=True)
    p3.add_argument("--rm-path", required=True)
    p3.add_argument("--test-n", type=int, default=50)
    p3.add_argument("--max-new-tokens", type=int, default=None)

    p4 = sub.add_parser("lm-eval", help="lm-evaluation-harness（可选）")
    p4.add_argument("--export-dir", default="checkpoints/hf")
    p4.add_argument("--tasks", default="ceval-valid")
    p4.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/eval.yaml", ROOT)
    tok = Tokenizer.load(cfg.tokenizer_path)

    if args.eval_mode == "ppl":
        mcfg = infer_model_config(args.model_path, mcfg, device)
        model = AhaForCausalLM(mcfg)
        load_pretrained(model, args.model_path, device)
        model = model.to(device).eval()
        ppl = evaluate_ppl(
            model, tok, args.ppl_file,
            max_seq_len=cfg.max_seq_len, batch_size=cfg.batch_size,
            device=device, max_samples=cfg.max_samples,
        )
        print(f"PPL = {ppl:.4f}")

    elif args.eval_mode == "samples":
        mcfg = infer_model_config(args.model_path, mcfg, device)
        model = AhaForCausalLM(mcfg)
        load_pretrained(model, args.model_path, device)
        model = model.to(device).eval()
        prompts = None
        if args.prompt_file:
            p = Path(args.prompt_file)
            prompts = []
            if p.suffix == ".jsonl":
                prompts = [item.get("prompt") or item.get("text") for item in read_jsonl(p)]
            else:
                prompts = [line.strip() for line in open(p, encoding="utf-8") if line.strip()]
        generate_samples(
            model, tok, prompts,
            max_new_tokens=args.max_new_tokens or cfg.max_new_tokens,
            temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
            do_sample=cfg.do_sample, device=device, chat=args.chat,
        )

    elif args.eval_mode == "compare":
        sft_cfg = infer_model_config(args.sft_path, mcfg, device)
        rl_cfg = infer_model_config(args.rl_path, mcfg, device)
        rm_cfg = infer_model_config(args.rm_path, mcfg, device)
        sft_model = AhaForCausalLM(sft_cfg)
        rl_model = AhaForCausalLM(rl_cfg)
        rm = RewardModel(rm_cfg)
        load_pretrained(sft_model, args.sft_path, device)
        load_pretrained(rl_model, args.rl_path, device)
        load_pretrained(rm, args.rm_path, device, strict=False)
        sft_model = sft_model.to(device).eval()
        rl_model = rl_model.to(device).eval()
        rm = rm.to(device).eval()
        dpo_items = read_jsonl(Path(cfg.data_dir) / "dpo.jsonl", max_lines=args.test_n)
        result = compare_models(
            sft_model, rl_model, rm, tok, dpo_items,
            max_new_tokens=args.max_new_tokens or cfg.max_new_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p, device=device,
        )
        out = Path(cfg.output_dir) / f"compare_{cfg.run_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {out}")

    elif args.eval_mode == "lm-eval":
        from ahamodel.eval.benchmark import run_lm_eval

        run_lm_eval(args.export_dir, tasks=args.tasks, limit=args.limit)


if __name__ == "__main__":
    main()
