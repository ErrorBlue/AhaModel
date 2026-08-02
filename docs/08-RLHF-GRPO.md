# 08 RLHF-GRPO：无 critic 的组内归一化

## 1. 为什么 GRPO

PPO 的 critic 又大又难训，且价值估计偏差影响稳定性。DeepSeek-R1 等采用 GRPO：
对同一个 prompt 采样一组回答，用**组内相对优势**替代 critic：

```
A_i = (r_i - mean(G)) / std(G)
```

回答只要比组内其他回答好就是正优势，不需要绝对值准确。

## 2. 规则奖励（RLVR 思路）

现代 RL 用「可验证信号」替代 RM：数学答案匹配、代码单测、格式校验。
本项目 `ahamodel/data/math_qa.py` 内置小学算术题生成器：

```text
prompt : 请计算：123 + 456 = ? 请只输出答案数字。
reward : 提取回复中最后一个整数，与答案相等得 1 分，否则 0 分
```

奖励函数是插拔式的：想换任务只改 `reward_type` 与规则函数即可。

## 3. GRPO 损失

```
loss = -E[ min(ratio, clip(ratio, 1-ε, 1+ε)) · A ] - β · KL(π_ref || π)
```

无 value loss、无 GAE，训练比 PPO 轻很多。KL 这里作为正则项直接扣在损失里
（代码用 k2 无偏估计，注释说明 k3 变体）。

## 4. 运行

```bash
python scripts/09_grpo.py --model-path checkpoints/sft/latest.pt
```

`configs/grpo.yaml` 关键参数：`group_size=8`（每 prompt 采样数）、`kl_coef=0.04`、
`reward_type=math`。看 `reward`（组内平均规则得分）是否随训练上升。

## 5. 与 PPO 的对比实验建议

同一 SFT 起点、同 batch 预算，分别跑 PPO 与 GRPO，观察：

- 训练耗时与显存（GRPO 省掉 critic）；
- 数学题正确率上升曲线；
- 输出多样性（KL 惩罚相同的前提下）。

## Smoke 测试与正式运行

- **Smoke（快速验证）**：`python scripts/09_grpo.py --smoke --run-name grpo-smoke --model-path checkpoints/sft/latest.pt`
- **正式（4090）**：`python scripts/09_grpo.py --model-path checkpoints/sft/latest.pt --use-logger swanlab`

**阶段产物**：`checkpoints/grpo/latest.pt`（GRPO 训练后的策略权重）。

## 扩展思路

- 把规则奖励换成代码执行（`exec` 沙箱）验证输出；
- 接入真实数学数据集（如 GSM8K 子集）做 RLVR；
- 实现 CISPO / 带 reference-free 的变体。
