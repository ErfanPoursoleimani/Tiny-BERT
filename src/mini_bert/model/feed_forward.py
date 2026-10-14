"""Position-wise feed-forward network: FFN(x) = W2 GELU(W1 x + b1) + b2."""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.act = nn.GELU()  # exact (erf) GELU as in original BERT
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
