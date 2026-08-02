# 06 RLHF-DPO：直接偏好优化

## 1. 从 SFT 到偏好优化

SFT 只能模仿「好回答的样子」，学不到「什么更好」。偏好优化用 `(prompt, chosen, rejected)`
三元组，让模型把 chosen 的概率推高、rejected 压低。

## 2. DPO 公式

DPO 不需要单独训练奖励模型（这正是它比 PPO 简单的原因）。设 π 是当前策略、
π_ref 是冻结的 SFT 模型，β 是温度：

```
loss = -log σ( β * [ (logπ(yw|x) - logπ_ref(yw|x))
                  - (logπ(yl|x) - logπ_ref(yl|x)) ] )
```

直觉：chosen 相对 reference 提升越多，损失越小；rejected 相对 reference 提升会被惩罚。
其中 `logπ(y|x)` 是整段回答的 log 概率之和。

## 3. 代码导读

- `response_logprobs`（`ahamodel/train/rl.py`）：用 labels 掩码只取响应段 token 的
  log-softmax 概率并求和；
- `DpoDataset`：chosen/rejected 各自编码为 `(ids, labels, response_start)`；
- `scripts/06_dpo.py`：reference 深拷贝自策略模型并冻结，每步算 4 个 logp 后套公式。

数据用 `dpo.jsonl`（`{"chosen": [...], "rejected": [...]}` 对话格式）。

## 4. 运行

```bash
python scripts/06_dpo.py --model-path checkpoints/sft/latest.pt
```

默认 `beta=0.1, lr=1e-5`（对齐阶段 lr 要比 SFT 小一个量级）。日志里的 `dpo_acc`
表示 chosen 隐式奖励高于 rejected 的比例，越高越好。

## 5. 调参

- **β 太大**：过拟合偏好数据，输出退化；**β 太小**：几乎没有对齐效果；
- **chosen/rejected 长度差异**：logp 按整段求和，长度差异过大会干扰，可按长度分桶；
- **数据质量**：偏好数据噪声比数量更重要，先人工看 100 条。

## Smoke 测试与正式运行

- **Smoke（快速验证）**：`python scripts/06_dpo.py --smoke --run-name dpo-smoke --model-path checkpoints/sft/latest.pt`
- **正式（4090）**：`python scripts/06_dpo.py --model-path checkpoints/sft/latest.pt --use-logger swanlab`

**阶段产物**：`checkpoints/dpo/latest.pt`（偏好对齐后的策略权重）。

## 扩展思路

- 实现 IPO/SimPO 对比（改公式一行即可）；
- 用 DPO 数据顺便训练 RM（本项目 scripts/07_rm.py 就是这么做的），
  体会「同数据、不同目标」的两种算法。
