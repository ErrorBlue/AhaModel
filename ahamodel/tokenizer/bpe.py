"""字节级 BPE（Byte-level BPE）从零实现。

设计：
- 特殊 token 占用 id 0..S-1（S 为特殊 token 数量）
- 原始字节占用 id S..S+255
- 每次合并产生一个新 id，从 S+256 开始递增
- 训练使用「出现位置表 + 双向链表」增量更新出现次数，
  避免每轮合并都对整个语料做一次全量扫描

参考：minimind 的 train_tokenizer / tiktoken 的字节级 BPE 设计。
"""

from __future__ import annotations

import heapq
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class BPE:
    def __init__(self, special_tokens: Sequence[str] = (), max_vocab_size: int = 6400):
        self.special_tokens: List[str] = list(special_tokens)
        self.max_vocab_size: int = max_vocab_size
        self.num_specials: int = len(self.special_tokens)
        self.byte_start: int = self.num_specials
        self.byte_end: int = self.num_specials + 256
        self.merges: List[Tuple[int, int, int]] = []  # (a, b, new_id)，按下标即训练顺序（rank）
        self.merge_rank: Dict[Tuple[int, int], int] = {}
        self.merge_map: Dict[Tuple[int, int], int] = {}
        self.id_to_bytes: Dict[int, bytes] = {}
        self.vocab_size: int = self.num_specials + 256

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(
        self,
        texts: Iterable[str],
        max_vocab_size: Optional[int] = None,
        min_freq: int = 2,
        verbose: bool = True,
    ) -> "BPE":
        """在语料上训练 BPE。

        texts: 迭代器，每个元素是一段文本。
        max_vocab_size: 目标词表大小（上限）；默认用构造时的值。
        min_freq: 当最佳 pair 出现次数低于该值时停止合并。
        """
        max_vocab_size = max_vocab_size or self.max_vocab_size
        if self.num_specials + 256 >= max_vocab_size:
            raise ValueError("max_vocab_size 必须大于 特殊token数+256")

        # 1) 语料 -> 字节 id 序列
        seqs: List[List[int]] = []
        for text in texts:
            text = text or ""
            if text:
                seqs.append([self.byte_start + b for b in text.encode("utf-8")])
        if not seqs:
            raise ValueError("语料为空，无法训练 tokenizer")

        # 2) 双向链表（数组实现）：vals/prevs/nexts 三个数组构成一条链
        #    - vals[i] 当前 token id；prevs[i]/nexts[i] 前后节点下标，-1 表示无
        #    - 每个文本的链头固定是下标 0；合并时只改 nexts/prevs 指针，
        #      数组长度不变，因此“节点下标”在合并过程中保持稳定
        vals: List[List[int]] = []
        prevs: List[List[int]] = []
        nexts: List[List[int]] = []
        for seq in seqs:
            n = len(seq)
            vals.append(list(seq))
            prevs.append(list(range(-1, n - 1)))
            nxt = list(range(1, n + 1))
            nxt[-1] = -1
            nexts.append(nxt)

        # 3) 出现位置表：pair -> [(text_idx, node_idx), ...]
        #    counts[pair] 恒等于「有效出现次数」（增量维护，见合并循环）
        occ: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        counts: Counter = Counter()
        for t, vt in enumerate(vals):
            for i in range(len(vt) - 1):
                pair = (vt[i], vt[i + 1])
                occ.setdefault(pair, []).append((t, i))
                counts[pair] += 1

        # 4) 合并循环：每轮取出现次数最多的 pair，把语料中所有该 pair 合并为新 id
        next_id = self.byte_end
        merges: List[Tuple[int, int, int]] = []
        while next_id < max_vocab_size:
            if not counts:
                break
            (a, b), cnt = counts.most_common(1)[0]
            if cnt < min_freq:
                break

            new_id = next_id
            next_id += 1
            processed = 0
            for t, i in list(occ.get((a, b), ())):
                # 陈旧条目防御：位置可能已被其他合并改变
                j = nexts[t][i]
                if j == -1 or vals[t][i] != a or vals[t][j] != b:
                    continue
                p = prevs[t][i]
                n = nexts[t][j]

                # 删除被破坏的 pair 计数：
                #   1) 本出现 (i,j)=(a,b) 自身
                #   2) 左邻 (p,i)；3) 右邻 (j,n)
                counts[(a, b)] -= 1
                if p != -1:
                    counts[(vals[t][p], vals[t][i])] -= 1
                if n != -1:
                    counts[(vals[t][j], vals[t][n])] -= 1

                # 执行合并：节点 i 变为 new_id，跳过节点 j
                vals[t][i] = new_id
                nexts[t][i] = n
                if n != -1:
                    prevs[t][n] = i
                # 关键：把被移除的右节点 j 标记为哨兵，使链上已失效的
                # 陈旧出现记录无法通过有效性检查，避免重复扣减计数
                vals[t][j] = -1
                nexts[t][j] = -1

                # 新增 pair 计数：(p,i) 与 (i,n)
                if p != -1:
                    pair = (vals[t][p], vals[t][i])
                    counts[pair] += 1
                    # 出现记录一律指向 pair 的“左节点”：这里是 p
                    occ.setdefault(pair, []).append((t, p))
                if n != -1:
                    pair = (vals[t][i], vals[t][n])
                    counts[pair] += 1
                    occ.setdefault(pair, []).append((t, i))
                processed += 1

            if processed == 0:
                break
            merges.append((a, b, new_id))

            # 定期重建出现位置表，清除陈旧条目，控制内存与后续扫描成本
            if len(merges) % 500 == 0:
                occ = {}
                counts = Counter()
                for t in range(len(vals)):
                    i = 0
                    while i != -1:
                        j = nexts[t][i]
                        if j != -1:
                            pair = (vals[t][i], vals[t][j])
                            occ.setdefault(pair, []).append((t, i))
                            counts[pair] += 1
                        i = j

            if verbose and len(merges) % 1000 == 0:
                print(f"  BPE merge {len(merges)}: 词表 {self.num_specials + 256 + len(merges)}, "
                      f"当前最佳 pair {a}->{b} 次数 {cnt}")

        self.merges = merges
        self.merge_rank = {pair: idx for idx, (a, b, _) in enumerate(merges)}
        self.merge_map = {pair: new for a, b, new in merges}
        self.vocab_size = self.num_specials + 256 + len(merges)
        self._build_id_to_bytes()
        return self

    def _build_id_to_bytes(self) -> None:
        m: Dict[int, bytes] = {self.byte_start + b: bytes([b]) for b in range(256)}
        for a, b, new in self.merges:
            m[new] = m[a] + m[b]
        self.id_to_bytes = m

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def encode_bytes(self, data: bytes) -> List[int]:
        """把任意字节串编码成 token id 序列（不含特殊 token）。"""
        ids = [self.byte_start + b for b in data]
        return self._apply_merges(ids)

    def _apply_merges(self, ids: List[int]) -> List[int]:
        """按 merge rank 从小到大应用合并。

        用最小堆维护「相邻 pair 的 rank」，每次弹出 rank 最小的合法 pair 合并，
        只把受影响的邻接 pair 重新入堆，平均 O(n log n)。
        """
        if not self.merge_rank or len(ids) < 2:
            return ids
        heap: List[Tuple[int, int]] = []
        for i in range(len(ids) - 1):
            r = self.merge_rank.get((ids[i], ids[i + 1]))
            if r is not None:
                heapq.heappush(heap, (r, i))
        result = list(ids)
        while heap:
            r, i = heapq.heappop(heap)
            if i + 1 >= len(result):
                continue
            if self.merge_rank.get((result[i], result[i + 1])) != r:
                continue  # 陈旧堆条目
            new = self.merge_map[(result[i], result[i + 1])]
            result[i] = new
            del result[i + 1]
            if i - 1 >= 0:
                r2 = self.merge_rank.get((result[i - 1], result[i]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i - 1))
            if i + 1 < len(result):
                r2 = self.merge_rank.get((result[i], result[i + 1]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i))
        return result

    def decode_ids(self, ids: Sequence[int]) -> bytes:
        """把 token id 序列解码回原始字节（特殊 token 不含在内）。"""
        out = bytearray()
        for i in ids:
            if i < self.num_specials:
                raise ValueError(f"id {i} 是特殊 token，应由 Tokenizer 处理")
            out.extend(self.id_to_bytes.get(i, b""))
        return bytes(out)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
            "max_vocab_size": self.max_vocab_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPE":
        bpe = cls(special_tokens=d["special_tokens"], max_vocab_size=d["max_vocab_size"])
        bpe.merges = [tuple(m) for m in d["merges"]]
        bpe.merge_rank = {pair: idx for idx, pair in enumerate((m[0], m[1]) for m in bpe.merges)}
        bpe.merge_map = {(a, b): new for a, b, new in bpe.merges}
        bpe.vocab_size = bpe.num_specials + 256 + len(bpe.merges)
        bpe._build_id_to_bytes()
        return bpe
