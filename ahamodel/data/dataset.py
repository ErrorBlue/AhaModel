"""三阶段数据集 + collate 函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ahamodel.data.template import ChatTemplate


def read_jsonl(path: str | Path, max_lines: Optional[int] = None) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ----------------------------------------------------------------------
# 预训练
# ----------------------------------------------------------------------
class PretrainDataset(Dataset):
    """每行 {"text": ...}，返回 token 序列（截断到 max_seq_len）。"""

    def __init__(self, tokenizer, jsonl_path: str | Path, max_seq_len: int, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = [item["text"] for item in read_jsonl(jsonl_path, max_samples)]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int) -> List[int]:
        ids = self.tokenizer.encode(self.texts[i])
        return ids[: self.max_seq_len]


def collate_lm(batch: List[List[int]], pad_id: int) -> dict:
    """预训练 collate：右 padding，labels 为目标 token（pad 位置 -100）。"""
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(ids) - 1] = torch.tensor(ids[1:])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ----------------------------------------------------------------------
# SFT
# ----------------------------------------------------------------------
class SftDataset(Dataset):
    """每行 {"conversations": [...]}，返回 (input_ids, labels)，只对 assistant 算 loss。"""

    def __init__(self, tokenizer, jsonl_path: str | Path, max_seq_len: int, max_samples: Optional[int] = None):
        self.template = ChatTemplate(tokenizer, max_seq_len)
        self.items = read_jsonl(jsonl_path, max_samples)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[List[int], List[int]]:
        return self.template.encode_with_labels(self.items[i]["conversations"])


def collate_sft(batch: List[Tuple[List[int], List[int]]], pad_id: int) -> dict:
    input_ids, labels = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lab_t = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        ids_t[i, : len(ids)] = torch.tensor(ids)
        lab_t[i, : len(labs)] = torch.tensor(labs)
        mask[i, : len(ids)] = 1
    return {"input_ids": ids_t, "labels": lab_t, "attention_mask": mask}


# ----------------------------------------------------------------------
# DPO / RM
# ----------------------------------------------------------------------
class DpoDataset(Dataset):
    """每行 {"chosen": [messages], "rejected": [messages]}。

    返回 chosen/rejected 的 (ids, labels, response_start)。
    response_start: labels 中第一个非 -100 的位置（即响应开始）。
    """

    def __init__(self, tokenizer, jsonl_path: str | Path, max_seq_len: int, max_samples: Optional[int] = None):
        self.template = ChatTemplate(tokenizer, max_seq_len)
        self.items = read_jsonl(jsonl_path, max_samples)

    def __len__(self):
        return len(self.items)

    def _one(self, messages) -> Tuple[List[int], List[int], int]:
        ids, labels = self.template.encode_with_labels(messages)
        start = next((j for j, v in enumerate(labels) if v != -100), len(labels))
        return ids, labels, start

    def __getitem__(self, i: int):
        item = self.items[i]
        c = self._one(item["chosen"])
        r = self._one(item["rejected"])
        return (*c, *r)


def collate_dpo(batch, pad_id: int) -> dict:
    """把 chosen/rejected 分别 padding 成两批。"""
    c_ids, c_labels, c_starts, r_ids, r_labels, r_starts = zip(*batch)

    def pack(id_list, label_list, starts):
        max_len = max(len(x) for x in id_list)
        ids = torch.full((len(id_list), max_len), pad_id, dtype=torch.long)
        labs = torch.full((len(id_list), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(id_list), max_len), dtype=torch.long)
        for i, (ids_i, labs_i) in enumerate(zip(id_list, label_list)):
            ids[i, : len(ids_i)] = torch.tensor(ids_i)
            labs[i, : len(labs_i)] = torch.tensor(labs_i)
            mask[i, : len(ids_i)] = 1
        return ids, labs, mask, torch.tensor(starts)

    c_ids, c_labs, c_mask, c_start = pack(c_ids, c_labels, c_starts)
    r_ids, r_labs, r_mask, r_start = pack(r_ids, r_labels, r_starts)
    return {
        "chosen": {"input_ids": c_ids, "labels": c_labs, "attention_mask": c_mask, "response_start": c_start},
        "rejected": {"input_ids": r_ids, "labels": r_labs, "attention_mask": r_mask, "response_start": r_start},
    }


# ----------------------------------------------------------------------
# RL rollout
# ----------------------------------------------------------------------
class RolloutDataset(Dataset):
    """rlaif.jsonl：对话最后一条是空 assistant，编码为「续写提示词」。"""

    def __init__(self, tokenizer, jsonl_path: str | Path, max_seq_len: int, max_samples: Optional[int] = None):
        self.template = ChatTemplate(tokenizer, max_seq_len)
        self.items = read_jsonl(jsonl_path, max_samples)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> List[int]:
        return self.template.encode_rollout_prompt(self.items[i]["conversations"])


def collate_rollout(batch: List[List[int]]) -> List[List[int]]:
    """rollout 需要保留每行长度信息，直接返回 list（由 generate_batch 内部 padding）。"""
    return batch
