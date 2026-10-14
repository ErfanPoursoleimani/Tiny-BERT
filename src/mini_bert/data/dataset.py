"""BERT pretraining examples built from tokenized documents.

Each document is a list of sentences, each sentence a list of token ids.
MLM-only: pack consecutive sentences into one segment [CLS] ... [SEP].
NSP     : [CLS] A [SEP] B [SEP]; 50% B is the true continuation (label 0 = IsNext),
          50% B comes from a random other document (label 1 = NotNext).
Example construction is deterministic given (seed, index); masking is NOT done here (see collator).
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

IS_NEXT, NOT_NEXT = 0, 1


class PretrainingDataset(Dataset):
    def __init__(self, docs: list[list[list[int]]], cls_id: int, sep_id: int, max_seq_length: int,
                 nsp: bool = False, short_seq_prob: float = 0.1, seed: int = 0) -> None:
        self.docs = [d for d in docs if len(d) > 0]
        if nsp:
            self.docs = [d for d in self.docs if len(d) >= 2]
        if not self.docs:
            raise ValueError("no usable documents")
        self.cls_id, self.sep_id, self.max_len = cls_id, sep_id, max_seq_length
        self.nsp, self.short_seq_prob, self.seed = nsp, short_seq_prob, seed
        # one "anchor" per (document, start sentence) so every sentence is used as a start point
        self.anchors = [(di, si) for di, d in enumerate(self.docs) for si in range(len(d))]
        if nsp:
            self.anchors = [(di, si) for di, si in self.anchors if si < len(self.docs[di]) - 1]

    def __len__(self) -> int:
        return len(self.anchors)

    def _pack(self, doc, start, budget):
        out, i = [], start
        while i < len(doc) and len(out) + len(doc[i]) <= budget:
            out.extend(doc[i]); i += 1
        if not out:  # single sentence longer than budget -> truncate
            out = doc[start][:budget]; i = start + 1
        return out, i

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed * 1_000_003 + idx)
        di, si = self.anchors[idx]
        doc = self.docs[di]
        budget = self.max_len - (3 if self.nsp else 2)
        if rng.random() < self.short_seq_prob:
            budget = rng.randint(max(8, budget // 4), budget)
        if not self.nsp:
            a, _ = self._pack(doc, si, budget)
            ids = [self.cls_id] + a + [self.sep_id]
            return {"input_ids": ids, "token_type_ids": [0] * len(ids), "next_sentence_label": -1}
        a, nxt = self._pack(doc, si, max(1, budget // 2))
        label = IS_NEXT
        if nxt >= len(doc) or rng.random() < 0.5:
            label = NOT_NEXT
        if label == IS_NEXT:
            b, _ = self._pack(doc, nxt, budget - len(a))
        else:
            if len(self.docs) > 1:
                dj = rng.randrange(len(self.docs) - 1)
                dj += dj >= di  # guarantee a different document
            else:
                dj = di
            other = self.docs[dj]
            b, _ = self._pack(other, rng.randrange(len(other)), budget - len(a))
        a, b = a[: budget - 1], b[: max(1, budget - len(a[: budget - 1]))]
        ids = [self.cls_id] + a + [self.sep_id] + b + [self.sep_id]
        types = [0] * (len(a) + 2) + [1] * (len(b) + 1)
        return {"input_ids": ids, "token_type_ids": types, "next_sentence_label": label}
