"""Linear warmup then linear (or cosine) decay from max LR to min LR."""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(opt: Optimizer, warmup: int, total: int, lr: float, min_lr: float, kind: str = "linear"):
    floor = min_lr / lr

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)  # 0 -> 1 (never exactly 0 at the first step)
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        if kind == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
        if kind == "linear":
            return 1.0 - (1.0 - floor) * prog
        raise ValueError(f"unknown scheduler {kind}")

    return LambdaLR(opt, f)
