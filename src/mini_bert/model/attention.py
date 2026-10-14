"""Bidirectional multi-head self-attention (NO causal mask).

Shapes (B=batch, T=seq, H=hidden, h=heads, d=H/h):
    x            : (B, T, H)
    Q, K, V      : (B, h, T, d)   after projection + head split
    scores       : (B, h, T, T) = Q K^T / sqrt(d) + M
    weights      : softmax over last dim
    context      : (B, h, T, d) -> (B, T, H) -> W_O
M is an additive padding mask (0 for real tokens, -inf-ish for [PAD]) of shape (B,1,1,T).
There is deliberately no triangular term: every position can attend to every non-pad position.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig


def make_padding_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B,T) of {1,0} -> additive (B,1,1,T): 0 where attend, large negative where pad."""
    neg = torch.finfo(dtype).min
    return (1.0 - attention_mask[:, None, None, :].to(dtype)) * neg


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.h = cfg.num_attention_heads
        self.d = cfg.head_dim
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # W_O
        self.attn_dropout_p = cfg.attention_probs_dropout_prob
        self.attn_dropout = nn.Dropout(cfg.attention_probs_dropout_prob)
        self.implementation = cfg.attention_implementation

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, hid = x.shape
        q, k, v = self._split(self.query(x)), self._split(self.key(x)), self._split(self.value(x))

        impl = self.implementation
        if impl == "auto":
            impl = "manual" if output_attentions else "sdpa"
        if output_attentions and impl == "sdpa":
            impl = "manual"  # SDPA cannot return weights

        weights = None
        if impl == "manual":
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
            if attention_mask is not None:
                scores = scores + make_padding_mask(attention_mask, scores.dtype)
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            probs = self.attn_dropout(weights)
            ctx = probs @ v
        else:
            bool_mask = None
            if attention_mask is not None:
                bool_mask = attention_mask[:, None, None, :].bool()
            ctx = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bool_mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,  # BERT is bidirectional
            )
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, hid)
        return self.out(ctx), (weights if output_attentions else None)
