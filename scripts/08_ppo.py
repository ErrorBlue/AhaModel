"""步骤 8：PPO（actor + reference + RM + critic/GAE）。"""

import argparse
import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ahamodel.data.dataset import RolloutDataset
from ahamodel.model.model import AhaForCausalLM
from ahamodel.tokenizer import Tokenizer
from ahamodel.train.ppo import CriticModel, PPOTrainer
from ahamodel.train.reward_model import RewardModel
from ahamodel.train.trainer import Logger
from ahamodel.utils.cli import add_common_args, infer_model_config, load_pretrained, setup_stage


def main():
    parser = argparse.ArgumentParser(description="PPO")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="SFT checkpoint（actor 初始）")
    parser.add_argument("--rm-path", default=None, help="奖励模型 checkpoint（scripts/07_rm.py 产物）")
    args = parser.parse_args()
    cfg, mcfg, device = setup_stage(args, "configs/ppo.yaml", ROOT)
    actor_path = args.model_path or cfg.model_path
    rm_path = args.rm_path or cfg.rm_path
    mcfg = infer_model_config(actor_path, mcfg, device)

    tok = Tokenizer.load(cfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size
    actor = AhaForCausalLM(mcfg)
    load_pretrained(actor, actor_path, device)
    actor = actor.to(device)

    reference = copy.deepcopy(actor).to(device).eval()
    for p in reference.parameters():
        p.requires_grad = False

    rm = RewardModel(mcfg).to(device)
    load_pretrained(rm, rm_path, device, strict=False)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False

    critic = CriticModel(mcfg).to(device)
    critic.model.load_state_dict(actor.state_dict())  # 用 actor 骨干初始化 critic

    data_file = Path(cfg.data_dir) / "rlaif.jsonl"
    if not data_file.exists():
        sys.exit(f"缺少 {data_file}，请先运行 scripts/01_prepare_data.py --stage rlaif")
    ds = RolloutDataset(tok, data_file, cfg.max_seq_len, cfg.max_samples)
    prompts = [ds[i] for i in range(min(len(ds), cfg.n_rollout_prompts))]
    print(f"rollout 提示词数量: {len(prompts)}")

    ppo = PPOTrainer(actor, critic, rm, reference, cfg, device, Logger(cfg.use_logger, cfg.run_name, cfg.output_dir))
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in range(cfg.max_steps):
        batch_prompts = random.sample(prompts, min(cfg.batch_size, len(prompts)))
        responses = ppo.rollout(batch_prompts, tok)
        metrics = ppo.step_once(batch_prompts, responses, tok)
        if (step + 1) % cfg.log_every == 0:
            print(f"[ppo] step {step+1}/{cfg.max_steps} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            ppo.logger.log(metrics, step + 1)
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ppo.save(run_dir / f"step_{step+1}.pt")
            ppo.save(run_dir / "latest.pt")
    ppo.save(run_dir / "latest.pt")
    ppo.logger.finish()
    print(f"PPO 完成，checkpoint: {run_dir}/latest.pt")


if __name__ == "__main__":
    main()
