"""BERT dynamic masking: select ~15% of eligible tokens; 80% [MASK] / 10% random / 10% unchanged."""
from __future__ import annotations

import torch

from ..config import MLMConfig


def mask_tokens(
    input_ids: torch.Tensor,
    special_ids: set[int] | list[int],
    mask_id: int,
    vocab_size: int,
    cfg: MLMConfig,
    generator: torch.Generator | None = None,
    num_reserved: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (masked_input_ids, labels). labels == -100 for non-selected positions.

    Never selects ids in `special_ids` ([PAD],[UNK],[CLS],[SEP],[MASK]). [UNK] is excluded because
    its original identity is unknowable. Random replacements are drawn from non-special ids
    [num_reserved, vocab_size).
    """
    ids = input_ids.clone()
    labels = input_ids.clone()
    eligible = torch.ones_like(ids, dtype=torch.bool)
    for s in special_ids:
        eligible &= ids != s
    rand = torch.rand(ids.shape, generator=generator)
    selected = (rand < cfg.probability) & eligible
    # guarantee at least one masked token per sequence that has eligible tokens
    # (avoids empty-loss batches on tiny sequences)
    no_sel = (~selected.any(dim=1)) & eligible.any(dim=1)
    if no_sel.any():
        scores = torch.rand(ids.shape, generator=generator).masked_fill(~eligible, -1.0)
        pick = scores.argmax(dim=1)
        rows = torch.nonzero(no_sel).squeeze(1)
        selected[rows, pick[rows]] = True
    labels[~selected] = -100

    r = torch.rand(ids.shape, generator=generator)
    to_mask = selected & (r < cfg.mask_probability)
    to_rand = selected & (r >= cfg.mask_probability) & (r < cfg.mask_probability + cfg.random_probability)
    ids[to_mask] = mask_id
    n = int(to_rand.sum())
    if n:
        rnd = torch.randint(num_reserved, vocab_size, (n,), generator=generator)
        ids[to_rand] = rnd
    return ids, labels
