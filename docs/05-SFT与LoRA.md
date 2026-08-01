# 05 SFT 与 LoRA：指令微调

## 1. 什么是 SFT

预训练模型只会续写，不会「听指令」。SFT 用 `(指令, 回答)` 数据把模型调成对话助手。
核心技巧是 **loss masking**：只对 assistant 回答计算损失，system/user 内容不参与。

## 2. 聊天模板（`ahamodel/data/template.py`）

```text
<|system|>你是助手<|user|>你好<|assistant|>你好！有什么可以帮你？<|eos|>
```

`encode_with_labels` 返回 `(input_ids, labels)`：`labels[t]` 是位置 t 要预测的目标，
非 assistant 内容位置为 -100。模板与 vLLM 端保持一致，训练/部署无分布偏移。

## 3. 全参 SFT

```bash
python scripts/04_sft.py --model-path checkpoints/pretrain/latest.pt
```

## 4. LoRA（从零实现，`ahamodel/train/lora.py`）

冻结原始权重 W，学习低秩增量：

```
W' = W + (A @ B) * (alpha / r)      # A: (in, r), B: (r, out)
```

- `r` 决定秩（表达能力），`alpha/r` 是缩放；
- 只训练 A/B，参数量从 64M 降到约 1~2M；
- `merge()` 把增量写回权重（部署时零额外开销），`unmerge()` 可还原；
- 与全参微调共用同一份数据与训练脚本，对比两者效果是经典教学实验。

```bash
python scripts/05_lora.py --model-path checkpoints/pretrain/latest.pt
# 产物: checkpoints/lora_merged.pt（已合并的干净权重）
```

## 5. 常见问题

- **对话只会续写、不遵循指令**：检查模板一致性、确认 loss 只在 assistant 段；
- **LoRA 训练完导出报 key 错误**：`scripts/11_export_hf.py` 会自动过滤 `.lora.` 结构键，
  或者用 `lora_merged.pt`；
- **r 越大越好？** 不一定，大 r 增加过拟合风险，16~64 是常见区间。

## 扩展思路

- 对比 LoRA 与全参 SFT 在固定评测集上的分数；
- 把 LoRA 应用到 attention 之外的层（如 norm 不适用，可试 embedding 增量）；
- 实现 QLoRA 思路：bf16 权重 + 分块量化，进一步省显存。
