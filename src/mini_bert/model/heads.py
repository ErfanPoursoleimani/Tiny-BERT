"""MLM head, pooler + NSP head, classification head."""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


class MLMHead(nn.Module):
    """H -> Linear -> GELU -> LayerNorm -> vocab projection (optionally tied to token embeddings)."""

    def __init__(self, cfg: ModelConfig, token_embedding_weight: nn.Parameter | None = None) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        if cfg.tie_embeddings and token_embedding_weight is not None:
            self.decoder.weight = token_embedding_weight  # shared parameter

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.layer_norm(self.act(self.dense(h)))) + self.bias


class Pooler(nn.Module):
    """[CLS] hidden state -> Linear -> Tanh."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.act = nn.Tanh()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(h[:, 0]))


class NSPHead(nn.Module):
    """Linear(H, 2). Label convention: 0 = IsNext (B follows A), 1 = NotNext (random B)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.classifier = nn.Linear(cfg.hidden_size, 2)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)
