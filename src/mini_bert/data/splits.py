"""Document-level splits (no sentence of one document appears in two splits -> no leakage)."""
from __future__ import annotations

import random


def split_documents(docs: list, ratios=(0.9, 0.05, 0.05), seed: int = 0) -> dict[str, list]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    n_tr = int(len(docs) * ratios[0])
    n_va = int(len(docs) * ratios[1])
    pick = lambda ids: [docs[i] for i in ids]  # noqa: E731
    return {"train": pick(idx[:n_tr]), "validation": pick(idx[n_tr:n_tr + n_va]), "test": pick(idx[n_tr + n_va:])}
