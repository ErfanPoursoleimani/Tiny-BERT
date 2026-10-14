"""Pads examples and applies dynamic masking per batch (new mask every time a sample is drawn)."""
from __future__ import annotations

import torch

from ..config import MLMConfig
from .masking import mask_tokens


class PretrainingCollator:
    def __init__(self, pad_id: int, mask_id: int, vocab_size: int, special_ids, mlm_cfg: MLMConfig,
                 nsp: bool = False, seed: int | None = None) -> None:
        self.pad_id, self.mask_id, self.vocab_size = pad_id, mask_id, vocab_size
        self.special_ids, self.mlm_cfg, self.nsp = set(special_ids), mlm_cfg, nsp
        self.gen = None
        if seed is not None:  # deterministic masking stream (evaluation / reproducibility)
            self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        t = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), t), self.pad_id, dtype=torch.long)
        types = torch.zeros((len(batch), t), dtype=torch.long)
        attn = torch.zeros((len(batch), t), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            types[i, :n] = torch.tensor(b["token_type_ids"])
            attn[i, :n] = 1
        masked, labels = mask_tokens(ids, self.special_ids, self.mask_id, self.vocab_size, self.mlm_cfg, self.gen)
        out = {"input_ids": masked, "token_type_ids": types, "attention_mask": attn, "labels": labels}
        if self.nsp:
            out["next_sentence_label"] = torch.tensor([b["next_sentence_label"] for b in batch])
        return out
