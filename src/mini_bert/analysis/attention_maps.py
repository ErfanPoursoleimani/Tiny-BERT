"""Attention heatmaps (exploratory only - a pretty pattern is not proof of a linguistic function)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_head(attn, tokens: list[str], layer: int, head: int, path: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(tokens) * 0.5), max(4, len(tokens) * 0.5)))
    ax.imshow(attn[layer, head].numpy(), cmap="viridis", vmin=0)
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    ax.set_xlabel("key (attended to)"); ax.set_ylabel("query")
    ax.set_title(f"layer {layer + 1} head {head + 1}")
    fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110); plt.close(fig)
