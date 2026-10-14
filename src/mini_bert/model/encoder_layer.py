"""Original BERT (post-LayerNorm) encoder layer.

x -> MHSA -> dropout -> +x -> LayerNorm -> FFN -> dropout -> +(.) -> LayerNorm
(Not GPT-2's pre-LN.)
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .attention import MultiHeadSelfAttention
from .feed_forward import FeedForward


class BertEncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(cfg)
        self.attn_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(cfg)
        self.ffn_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x, attention_mask=None, output_attentions: bool = False):
        a, w = self.attention(x, attention_mask, output_attentions)
        x = self.attn_norm(x + self.attn_dropout(a))
        x = self.ffn_norm(x + self.ffn_dropout(self.ffn(x)))
        return x, w
