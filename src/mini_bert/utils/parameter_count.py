"""Parameter counting by component (shared/tied parameters counted once)."""
from __future__ import annotations

from torch import nn


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p)); total += p.numel()
    return total


def breakdown(model) -> dict[str, int]:
    """Component breakdown for BertForPreTraining."""
    emb = model.bert.embeddings
    layers = model.bert.encoder.layers
    out = {
        "Token embeddings": emb.token_embeddings.weight.numel(),
        "Position embeddings": emb.position_embeddings.weight.numel(),
        "Token-type embeddings": emb.token_type_embeddings.weight.numel(),
        "Attention": sum(count_params(l.attention) for l in layers),
        "Feed-forward": sum(count_params(l.ffn) for l in layers),
        "LayerNorm": sum(count_params(m) for l in layers for m in (l.attn_norm, l.ffn_norm)) + count_params(emb.layer_norm),
    }
    seen = {id(model.bert.embeddings.token_embeddings.weight)}
    mlm = sum(p.numel() for p in model.mlm_head.parameters() if id(p) not in seen)
    out["MLM head (excl. tied decoder)"] = mlm
    out["NSP head (+pooler)"] = (count_params(model.nsp_head) + count_params(model.pooler)) if model.nsp_head else 0
    return out
