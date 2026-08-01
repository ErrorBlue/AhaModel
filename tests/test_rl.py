"""RL 工具测试：GAE / KL / logprob / DPO 损失方向。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("torch")
import torch

from ahamodel.config import ModelConfig
from ahamodel.model.model import AhaForCausalLM
from ahamodel.train.rl import compute_gae, make_response_labels, per_token_kl, response_logprobs


def test_gae_handcrafted():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    values = torch.tensor([[0.5, 1.0, 0.5]])
    masks = torch.ones_like(rewards, dtype=torch.long)
    adv, ret = compute_gae(rewards, values, masks, gamma=0.99, lam=0.95)
    assert abs(float(adv[0, 2]) - 1.5) < 1e-5
    assert abs(float(adv[0, 1]) - 0.90575) < 1e-5
    assert torch.allclose(adv + values, ret, atol=1e-6)


def test_gae_padding_zeroed():
    rewards = torch.tensor([[1.0, 0.0, 2.0, 0.0]])
    values = torch.tensor([[0.5, 1.0, 0.5, 0.0]])
    masks = torch.tensor([[1, 1, 1, 0]])
    adv, ret = compute_gae(rewards, values, masks, gamma=0.99, lam=0.95)
    assert float(adv[0, 3]) == 0.0 and float(ret[0, 3]) == 0.0


def test_kl_zero_when_identical():
    lp = torch.zeros(2, 4)
    assert torch.allclose(per_token_kl(lp, lp), torch.zeros(2, 4), atol=1e-6)
    assert (per_token_kl(torch.zeros(2, 4), torch.full((2, 4), -1.0)) > 0).all()


def test_response_logprobs_and_labels():
    torch.manual_seed(0)
    model = AhaForCausalLM(
        ModelConfig(vocab_size=300, d_model=64, n_layers=2, q_heads=4, kv_heads=2, intermediate_size=128, max_seq_len=64)
    ).eval()
    ids = torch.randint(1, 300, (2, 16))
    starts = torch.tensor([5, 6])
    lens = torch.tensor([8, 7])
    labels = make_response_labels(ids, starts, lens, pad_id=0)
    # 行 0 响应从位置 4 开始；行 1 从位置 5 开始
    assert labels[0, 4] != -100 and labels[0, 12] == -100
    assert labels[1, 5] != -100 and labels[1, 12] == -100
    assert (labels[:, -1] == -100).all()
    total, token_lp = response_logprobs(model, ids, labels)
    assert total.shape == (2,)
    assert token_lp.shape == ids.shape
    # 响应位置 logprob 之和等于 total
    per_sample = (token_lp * (labels != -100)).sum(-1)
    assert torch.allclose(total, per_sample, atol=1e-6)


def test_dpo_loss_formula_direction():
    import torch.nn.functional as F

    def dpo_loss(lp_c, lp_r, ref_c, ref_r, beta):
        inner = (lp_c - lp_r) - (ref_c - ref_r)
        return -F.logsigmoid(beta * inner).mean()

    # chosen 优于 rejected：模型对 chosen 的 logp 越高，损失越低
    bad = dpo_loss(torch.tensor([-0.5]), torch.tensor([-0.5]), torch.tensor([-0.5]), torch.tensor([-0.5]), 0.1)
    good = dpo_loss(torch.tensor([-0.2]), torch.tensor([-0.8]), torch.tensor([-0.5]), torch.tensor([-0.5]), 0.1)
    assert good < bad
    # ref 已偏袒 chosen 时，约束更容易满足，损失更低
    ref_bad = dpo_loss(torch.tensor([-0.5]), torch.tensor([-0.5]), torch.tensor([-0.8]), torch.tensor([-0.2]), 0.1)
    assert ref_bad < bad
