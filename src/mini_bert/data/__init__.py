"""Data pipeline: load_processed() reads token ids produced by scripts/prepare_data.py."""
from __future__ import annotations

import os

import torch

from .collator import PretrainingCollator
from .dataset import PretrainingDataset


def load_processed(data_dir: str, split: str) -> list:
    path = os.path.join(data_dir, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run scripts/prepare_data.py first")
    return torch.load(path)["docs"]


__all__ = ["PretrainingCollator", "PretrainingDataset", "load_processed"]
