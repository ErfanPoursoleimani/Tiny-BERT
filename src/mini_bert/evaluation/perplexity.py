"""Pseudo-perplexity for MLM (Salazar et al., 2020).

For each sequence, mask ONE token at a time (replace with [MASK], 100%), record -log p(true token | rest),
average over all tokens of all sequences, and exponentiate. This needs T forward passes per sequence, so it
is expensive; it is NOT comparable to autoregressive (GPT-style) perplexity, which factorises the joint
probability of the sequence. Treat it as a heuristic fluency score.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def pseudo_perplexity(model, sequences: list[list[int]], special_ids: set[int], mask_id: int, device,
                      max_tokens: int = 2000) -> float:
    model.eval()
    total, n = 0.0, 0
    for ids in sequences:
        pos = [i for i, t in enumerate(ids) if t not in special_ids]
        if not pos:
            continue
        x = torch.tensor(ids, device=device).unsqueeze(0).repeat(len(pos), 1)
        for r, p in enumerate(pos):
            x[r, p] = mask_id
        logits = model(x).mlm_logits  # (len(pos), T, V)
        sel = logits[torch.arange(len(pos)), torch.tensor(pos)]
        tgt = torch.tensor([ids[p] for p in pos], device=device)
        total += float(F.cross_entropy(sel.float(), tgt, reduction="sum")); n += len(pos)
        if n >= max_tokens:
            break
    return math.exp(total / max(n, 1))
