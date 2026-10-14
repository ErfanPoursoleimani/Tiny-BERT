"""BERT input embeddings: E = LN(E_token + E_position + E_segment), then dropout."""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


class BertEmbeddings(nn.Module):
    """Token + learned position + token-type embeddings.

    Input  : input_ids (B, T), optional token_type_ids (B, T)
    Output : (B, T, H)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(cfg.max_position_embeddings).unsqueeze(0), persistent=False
        )
        self.max_len = cfg.max_position_embeddings

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds max_position_embeddings {self.max_len}")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        pos = self.position_ids[:, :t]
        x = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(pos)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(x))
