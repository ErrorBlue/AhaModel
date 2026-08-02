# 07 RLHF-PPO：带奖励模型的强化学习

## 1. PPO 组件

PPO 是 on-policy RL，需要四个模型 + 一个值函数：

```text
actor    : 要训练的对话模型（从 SFT 初始化）
reference: actor 的冻结副本（算 KL 惩罚，防止跑偏）
reward   : RM 打分（scripts/07_rm.py 训练）
critic   : 值函数 V(s)，GAE 需要
```

## 2. 奖励模型（`ahamodel/train/reward_model.py`）

与 DPO 用同一份偏好数据，但目标是学一个标量打分器：

```
loss = -log σ(r(chosen) - r(rejected))
```

实现：Transformer 骨干 + 响应段 hidden 平均池化 → 线性头。先训 RM 再训 PPO：

```bash
python scripts/07_rm.py --model-path checkpoints/sft/latest.pt
```

## 3. token 级奖励与 GAE

每个响应 token 的奖励：

```text
r_t = -β * KL(π_ref || π)                    # 每个 token 都有 KL 惩罚
r_last = r_last + RM(prompt + response)      # 最后一步加上 RM 打分
```

GAE 把稀疏的最终奖励回传给每个 token，公式：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t + γ·λ·A_{t+1}
```

## 4. PPO 更新（`ahamodel/train/ppo.py`）

```
ratio = exp(logπ_new - logπ_old)
policy_loss = -E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ]
value_loss   = MSE(V(s), returns)
```

- `clip_eps=0.2`：限制单步更新幅度，防止崩坏；
- `ppo_epochs`：同一批 rollout 复用次数；
- critic 用 actor 骨干初始化，`--smoke` 可快速验证整条链路。

## 5. 运行

```bash
python scripts/08_ppo.py \
  --model-path checkpoints/sft/latest.pt \
  --rm-path checkpoints/rm/latest.pt
```

rollout 数据来自 `rlaif.jsonl`（对话最后一条是空 assistant，模型续写回答）。
关注指标：`reward`（RM 平均分）、`kl`（与 reference 的距离）、`response_len`。

## 6. 常见问题

- **reward 涨但输出退化**：KL 惩罚太小，调大 `kl_coef`；
- **loss 波动大**：`clip_eps` 调小、`lr` 调小、rollout 数量加大；
- **RM 过拟合**：RM 训练与验证分开，PPO 阶段 RM 必须冻结。

## Smoke 测试与正式运行

RM（奖励模型）：
- **Smoke**：`python scripts/07_rm.py --smoke --run-name rm-smoke --model-path checkpoints/sft/latest.pt`
- **正式**：`python scripts/07_rm.py --model-path checkpoints/sft/latest.pt --use-logger swanlab`

PPO：
- **Smoke**：`python scripts/08_ppo.py --smoke --run-name ppo-smoke --model-path checkpoints/sft/latest.pt --rm-path checkpoints/rm/latest.pt`
- **正式**：`python scripts/08_ppo.py --model-path checkpoints/sft/latest.pt --rm-path checkpoints/rm/latest.pt --use-logger swanlab`

**阶段产物**：
- `checkpoints/rm/latest.pt`：奖励模型权重；
- `checkpoints/ppo/latest.pt`：PPO 训练后的 actor 权重（含 critic 状态）。

## 扩展思路

- 用 PPO 与 DPO 训同一批偏好数据，对比生成质量与训练稳定性；
- 加 entropy bonus 提高探索；
- 把 `k2` KL 换成 `k3`（ρlogρ - ρ + 1，GRPO 论文风格）对比。
