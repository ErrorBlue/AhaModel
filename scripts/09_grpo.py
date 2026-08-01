"""步骤 9：GRPO（无 critic，组内 advantage 归一化 + 规则奖励）。"""

import argparse
import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahamodel.data.math_qa import make_math_prompts
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.grpo import GRPOTrainer
from ahamodel.train.trainer import Logger
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, setup_stage


def main():
    parser = argparse.ArgumentParser(description="GRPO")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="SFT checkpoint")
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/grpo.yaml", ROOT)
    model_path = args.model_path or cfg.model_path
    mcfg = infer_model_config(model_path, mcfg, device)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size
    actor = AhaForCausalLM(mcfg)
    load_pretrained(actor, model_path, device)
    actor = actor.to(device)
    reference = copy.deepcopy(actor).to(device).eval()
    for p in reference.parameters():
        p.requires_grad = False

    prompts = make_math_prompts(cfg.n_rollout_prompts, seed=cfg.seed)
    print(f"数学规则奖励提示词: {len(prompts)} 条")
    grpo = GRPOTrainer(actor, reference, cfg, device, Logger(cfg.use_logger, cfg.run_name, cfg.output_dir))
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in range(cfg.max_steps):
        batch = random.sample(prompts, min(cfg.batch_size, len(prompts)))
        metrics = grpo.step_once(batch, tok)
        if (step + 1) % cfg.log_every == 0:
            print(f"[grpo] step {step+1}/{cfg.max_steps} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            grpo.logger.log(metrics, step + 1)
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            grpo.save(run_dir / f"step_{step+1}.pt")
            grpo.save(run_dir / "latest.pt")
    grpo.save(run_dir / "latest.pt")
    grpo.logger.finish()
    print(f"GRPO 完成，checkpoint: {run_dir}/latest.pt")


if __name__ == "__main__":
    main()
