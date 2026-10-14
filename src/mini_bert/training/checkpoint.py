"""Checkpoint save/load (model, optimizer, scheduler, scaler, step, RNG, config, tokenizer info)."""
from __future__ import annotations

import os
import shutil

import torch

from ..utils.seed import get_rng_state, set_rng_state


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, step: int, epoch: int, config: dict,
                    seed: int, best_val: float, tokenizer_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tok_info = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        import hashlib
        with open(tokenizer_path, "rb") as f:
            tok_info = {"path": tokenizer_path, "sha256": hashlib.sha256(f.read()).hexdigest()}
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step, "epoch": epoch, "config": config, "seed": seed,
        "rng_state": get_rng_state(), "best_val": best_val, "tokenizer": tok_info,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu",
                    restore_rng: bool = True) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if restore_rng and state.get("rng_state"):
        try:
            set_rng_state(state["rng_state"])
        except Exception:
            pass
    return state


def copy_checkpoint(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
