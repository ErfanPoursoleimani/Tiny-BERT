"""Training-curve plots from the metric history."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_training_plots(history: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tr = [h for h in history if h["type"] == "train"]
    va = [h for h in history if h["type"] == "val"]

    def one(name, series, ylabel, logy=False):
        if not any(len(x[1]) for x in series):
            return
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for label, (xs, ys) in [(s[0], s[1]) for s in series]:
            ax.plot(xs, ys, label=label)
        ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=110); plt.close(fig)

    x = lambda rows: [r["step"] for r in rows]  # noqa: E731
    col = lambda rows, k: [r[k] for r in rows]  # noqa: E731
    one("loss.png", [("train MLM", (x(tr), col(tr, "train_mlm_loss"))), ("val MLM", (x(va), col(va, "mlm_loss")))], "MLM cross-entropy")
    one("mlm_accuracy.png", [("top-1", (x(va), col(va, "mlm_acc"))), ("top-5", (x(va), col(va, "mlm_top5_acc")))], "masked-token accuracy")
    one("learning_rate.png", [("lr", (x(tr), col(tr, "lr")))], "learning rate")
    one("grad_norm.png", [("grad norm", (x(tr), col(tr, "grad_norm")))], "gradient norm (pre-clip)")
    one("tokens_per_sec.png", [("tokens/s", (x(tr), col(tr, "tokens_per_sec")))], "tokens / s")
    one("gpu_memory.png", [("allocated", (x(tr), col(tr, "gpu_allocated_mb"))), ("peak", (x(tr), col(tr, "gpu_peak_mb")))], "MB")
