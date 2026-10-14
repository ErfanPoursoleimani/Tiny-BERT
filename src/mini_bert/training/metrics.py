"""Masked-token metrics (only positions with label != -100)."""
from __future__ import annotations

import torch


@torch.no_grad()
def mlm_counts(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Returns counts so they can be summed over batches: n, top1_correct, top5_correct."""
    m = labels != -100
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "top1": 0, "top5": 0}
    sel = logits[m]                       # (n, V)
    tgt = labels[m]
    k = min(5, sel.size(-1))
    topk = sel.topk(k, dim=-1).indices    # (n, k)
    return {"n": n, "top1": int((topk[:, 0] == tgt).sum()), "top5": int((topk == tgt[:, None]).any(-1).sum())}


class Averager:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.cnt: dict[str, float] = {}

    def add(self, key: str, value: float, weight: float = 1.0) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value * weight
        self.cnt[key] = self.cnt.get(key, 0.0) + weight

    def mean(self, key: str) -> float:
        return self.sums.get(key, 0.0) / max(self.cnt.get(key, 0.0), 1e-12)
