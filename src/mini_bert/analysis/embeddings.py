"""Embedding extraction + PCA (numpy SVD; deterministic)."""
from __future__ import annotations

import numpy as np
import torch


def pca(x: np.ndarray, k: int = 2) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :k] * s[:k]


@torch.no_grad()
def extract(model, input_ids: torch.Tensor, token_type_ids=None) -> dict[str, torch.Tensor]:
    model.eval()
    h, _ = model.bert(input_ids.unsqueeze(0), None if token_type_ids is None else token_type_ids.unsqueeze(0))
    emb = model.bert.embeddings
    return {"token_embeddings": emb.token_embeddings.weight.detach().cpu(),
            "position_embeddings": emb.position_embeddings.weight.detach().cpu(),
            "contextual": h[0].cpu(), "cls": h[0, 0].cpu()}
