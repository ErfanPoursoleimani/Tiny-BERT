"""Sentence representations for probing ([CLS] or mean-pooled)."""
from __future__ import annotations

import torch


@torch.no_grad()
def sentence_repr(model, input_ids, token_type_ids, attention_mask, kind: str = "cls") -> torch.Tensor:
    model.eval()
    h, _ = model.bert(input_ids, token_type_ids, attention_mask)
    if kind == "cls":
        return h[:, 0]
    m = attention_mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)
