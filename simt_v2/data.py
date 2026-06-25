"""
Dataset + collation for English -> Telugu SiMT (v2).

Reads the same JSONL files as the original pipeline
(simult_mt/data/filtered/{train,val,test}.json), each line {"id","source","target"}.

For every example we record three token offsets in the flattened chat sequence:
    source_start : first English source token
    source_end   : one past the last English source token
    target_start : first Telugu target token (where generation begins)
These offsets drive the wait-k mask in masking.py.
"""

import json
import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = "Translate the text below to Telugu."


def compute_offsets(tokenizer, source_text):
    """(source_start, source_end, target_start) for the chat-formatted prompt."""
    sys_only = [{"role": "system", "content": SYSTEM_PROMPT}]
    with_src = sys_only + [{"role": "user", "content": source_text}]

    def n_tokens(msgs, add_gen):
        s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
        return len(tokenizer.encode(s, add_special_tokens=False))

    source_start = n_tokens(sys_only, False)
    source_end   = n_tokens(with_src, False)
    target_start = n_tokens(with_src, True)
    return source_start, source_end, target_start


class SiMTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=320):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": r["source"]},
            {"role": "assistant", "content": r["target"]},
        ]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = self.tok.encode(full, add_special_tokens=False)[: self.max_len]

        ss, se, ts = compute_offsets(self.tok, r["source"])
        labels = list(ids)
        for j in range(min(ts, len(labels))):   # supervise Telugu target only
            labels[j] = -100

        return {
            "input_ids": ids,
            "labels": labels,
            "source_start": ss,
            "source_end": se,
            "target_start": ts,
        }


def make_collate(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = L - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "pad_mask":       torch.tensor(attn, dtype=torch.long),
            "source_start":   torch.tensor([b["source_start"] for b in batch], dtype=torch.long),
            "source_end":     torch.tensor([b["source_end"] for b in batch], dtype=torch.long),
            "target_start":   torch.tensor([b["target_start"] for b in batch], dtype=torch.long),
        }
    return collate
