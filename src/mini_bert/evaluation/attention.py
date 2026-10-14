"""Extract attention weights: returns tensor (L, h, T, T) for one sequence."""
from __future__ import annotations

import torch


@torch.no_grad()
def get_attention(model, input_ids: torch.Tensor, token_type_ids=None) -> torch.Tensor:
    model.eval()  # dropout off
    out = model(input_ids.unsqueeze(0), None if token_type_ids is None else token_type_ids.unsqueeze(0),
                output_attentions=True)
    return torch.stack([a[0] for a in out.attentions]).float().cpu()
