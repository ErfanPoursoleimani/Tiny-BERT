"""GPU / CPU memory monitoring and rough memory estimation."""
from __future__ import annotations

import os

import torch


def gpu_memory_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0}
    mb = 2**20
    return {
        "allocated": torch.cuda.memory_allocated() / mb,
        "reserved": torch.cuda.memory_reserved() / mb,
        "peak_allocated": torch.cuda.max_memory_allocated() / mb,
    }


def cpu_ram_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return int(ln.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def estimate_memory_mb(n_params: int, param_bytes: int = 4, batch: int = 4, seq: int = 256,
                       hidden: int = 384, layers: int = 6, heads: int = 6, inter: int = 1536,
                       act_bytes: int = 2) -> dict[str, float]:
    """VERY ROUGH estimate (not exact VRAM). AdamW keeps two fp32 states per parameter (m, v), i.e. 2x the
    weight memory, plus fp32 gradients; so weights+grads+optimizer ~ 4x parameter bytes in fp32 training."""
    mb = 2**20
    weights = n_params * param_bytes
    grads = n_params * param_bytes
    optim = n_params * 4 * 2
    # per layer: ~ (qkv+out+ffn in/out+LN residuals) ~ 10*B*T*H + 2*B*T*I + attention scores/probs 2*B*h*T*T
    per_layer = (10 * batch * seq * hidden + 2 * batch * seq * inter + 2 * batch * heads * seq * seq) * act_bytes
    acts = per_layer * layers
    return {"weights": weights / mb, "gradients": grads / mb, "optimizer_states": optim / mb,
            "activations(approx)": acts / mb,
            "total(approx, excl. CUDA ctx/fragmentation)": (weights + grads + optim + acts) / mb}


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
