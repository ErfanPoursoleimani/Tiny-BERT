"""AdamW with weight-decay exclusion for biases and LayerNorm parameters."""
from __future__ import annotations

import torch
from torch import nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.999), eps=1e-8,
                    fused: bool = False) -> torch.optim.AdamW:
    decay, no_decay, seen = [], [], set()
    for mod_name, mod in model.named_modules():
        for pn, p in mod.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if pn.endswith("bias") or isinstance(mod, nn.LayerNorm) or pn == "bias":
                no_decay.append(p)
            else:
                decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"fused": True} if fused and torch.cuda.is_available() else {}
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, **kwargs)
