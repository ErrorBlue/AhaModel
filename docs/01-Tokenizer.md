# 01 Tokenizer：手写字节级 BPE

## 1. 为什么需要 tokenizer

模型吃的是数字，不是文字。Tokenizer 把文本变成 id 序列（编码），也能把 id 变回文本（解码）。
分词粒度有三种：

- **字符级**：每个字符一个 token。简单，但序列长、信息密度低；
- **子词级（BPE/SentencePiece）**：高频词整体成 token、低频拆成片段，主流 LLM 的选择；
- **词级**：词汇量爆炸，OOV 问题严重，几乎不用。

本项目实现**字节级 BPE**：先按 UTF-8 把文本拆成 256 种字节，再不断合并「出现最多的相邻对」。
中文生僻字、emoji 都能无 OOV 覆盖，这是 byte-level 的核心优势。

## 2. BPE 训练过程

假设语料里 `"低"` 和 `"频"` 相邻出现 100 次，是最频繁的 pair，就生成一个新 token `"低频"`，
词表 +1；重复该过程直到词表达到目标大小。合并顺序（rank）决定了编码规则：

```text
初始: [低, 频]  [数, 据]  [低, 频, 出, 现]
合并1(rank0): "低频"          -> 新 id 263
合并2(rank1): "数据"          -> 新 id 264
编码时按 rank 从小到大应用: [263] [264] [263, 出, 现]
```

## 3. 代码导读（`ahamodel/tokenizer/bpe.py`）

训练是核心难点。朴素做法每轮合并都全量扫描语料数 pair 次数，O(合并轮数 × 语料长度)，
6400 词表根本跑不动。本项目用两个数据结构做到近似线性：

- **双向链表（数组实现）**：每条文本一个链表，合并时只改前后指针，O(1) 删除节点；
- **出现位置表 `occ`**：`pair -> [(文本, 节点)]`，只遍历被合并 pair 的出现位置；
  合并一个位置时，只有左右两个邻居的 pair 计数会变，增量更新即可。

注意两个易错点（代码里都有注释）：

1. 被移除的右节点要打 `-1` 哨兵，否则陈旧记录会二次处理、计数被重复扣减；
2. 新增 pair 的出现记录必须指向**左节点**，指向右节点会导致记录永远失效。

编码（`_apply_merges`）用最小堆按 rank 弹出相邻 pair 合并，平均 O(n log n)。

## 4. 运行

```bash
# 用清洗后的语料训练（词表默认 6400）
python scripts/01_train_tokenizer.py --corpus data/pretrain.jsonl --vocab-size 6400

# 顺便导出 HF 格式（vLLM/评测要用）
python scripts/01_train_tokenizer.py --export-hf checkpoints/hf
```

产物 `data/tokenizer.json`：特殊 token（`<|pad|> <|unk|> <|bos|> <|eos|> <|system|> <|user|> <|assistant|>`）
占 0-6 号 id，字节占 7-262，合并 token 从 263 开始。

## 5. 调参建议

- **词表大小**：小模型建议 4000~8000。词表越大 embedding/输出层参数占比越高；
- **min_freq**：训练停在最佳 pair 出现次数 < min_freq（默认 2），语料太小时可改 1；
- **训练样本量**：`--sample 200000`（约几十 MB）足够，不必用全量语料训 tokenizer。

## 6. 常见问题

- **解码乱码**：UTF-8 多字节字符可能被拆到相邻 token，解码时先拼字节再统一 decode（代码已处理）；
- **模型词表和 tokenizer 不一致**：训练脚本会自动把 `model_cfg.vocab_size` 对齐到 tokenizer 实际词表。

## 扩展思路

- 用 `tokenizers` 库同语料训练并对比词表/压缩率；
- 尝试字符级分词跑一遍预训练，对比 PPL 与训练速度；
- 实现 SentencePiece 风格的 unigram 分词。
