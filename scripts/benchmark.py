"""Latency / throughput / memory benchmark and max-micro-batch search.

Only genuine CUDA out-of-memory errors are caught; every other exception propagates.
"""
import argparse
import json
import time

import _common  # noqa: F401
import torch

from mini_bert.builders import build_model
from mini_bert.config import load_config
from mini_bert.evaluation.mlm import autocast_ctx
from mini_bert.utils.device import select_device_and_precision
from mini_bert.utils.parameter_count import count_params
from mini_bert.utils.profiling import cpu_ram_mb, gpu_memory_mb, reset_peak


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def one_step(model, batch, seq, vocab, dev, dtype, scaler=None):
    ids = torch.randint(5, vocab, (batch, seq), device=dev)
    labels = torch.where(torch.rand(batch, seq, device=dev) < 0.15, ids, torch.full_like(ids, -100))
    labels[:, 0] = ids[:, 0]  # at least one target
    with autocast_ctx(dev, dtype):
        loss = model(ids, labels=labels).loss
    (scaler.scale(loss) if scaler else loss).backward()
    model.zero_grad(set_to_none=True)


def bench(cfg, batch, seq, precision, impl, ckpt, iters, warmup):
    cfg.model.attention_implementation = impl
    prec = select_device_and_precision(precision)
    dev = prec.device
    model = build_model(cfg).to(dev)
    model.bert.set_gradient_checkpointing(ckpt)
    model.train()
    vocab = cfg.model.vocab_size
    reset_peak()
    ids = torch.randint(5, vocab, (batch, seq), device=dev)
    labels = torch.full_like(ids, -100); labels[:, ::7] = ids[:, ::7]
    fwd = bwd = 0.0
    for i in range(warmup + iters):
        _sync(dev); t0 = time.perf_counter()
        with autocast_ctx(dev, prec.dtype):
            loss = model(ids, labels=labels).loss
        _sync(dev); t1 = time.perf_counter()
        loss.backward(); _sync(dev); t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        if i >= warmup:
            fwd += t1 - t0; bwd += t2 - t1
    fwd, bwd = fwd / iters, bwd / iters
    model.eval()
    with torch.no_grad():
        _sync(dev); t0 = time.perf_counter()
        for _ in range(iters):
            with autocast_ctx(dev, prec.dtype):
                model(ids[:1])
        _sync(dev); inf = (time.perf_counter() - t0) / iters
    mem = gpu_memory_mb()
    return {"precision": prec.name, "attention": impl, "grad_ckpt": ckpt, "batch": batch, "seq": seq,
            "params": count_params(model), "fwd_ms": fwd * 1e3, "bwd_ms": bwd * 1e3,
            "tokens_per_sec": batch * seq / (fwd + bwd), "samples_per_sec": batch / (fwd + bwd),
            "inference_ms(batch=1)": inf * 1e3, "peak_vram_mb": mem["peak_allocated"], "cpu_ram_mb": cpu_ram_mb()}


def find_max_batch(cfg, seq, precision, impl, ckpt, limit=256):
    """Doubling search then binary refinement; handles ONLY torch.cuda.OutOfMemoryError."""
    prec = select_device_and_precision(precision)
    if prec.device.type != "cuda":
        raise SystemExit("--find-max-batch-size needs a CUDA GPU (NOT EXECUTED in CPU-only environments)")
    cfg.model.attention_implementation = impl

    def ok(b):
        model = build_model(cfg).to(prec.device); model.bert.set_gradient_checkpointing(ckpt); model.train()
        try:
            reset_peak(); one_step(model, b, seq, cfg.model.vocab_size, prec.device, prec.dtype)
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            del model; torch.cuda.empty_cache()

    lo, b = 0, 1
    while b <= limit and ok(b):
        lo, b = b, b * 2
    hi = min(b, limit + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--find-max-batch-size", action="store_true")
    ap.add_argument("--out", default="outputs/metrics/benchmark.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seq = a.seq_len or cfg.model.max_position_embeddings
    if a.find_max_batch_size:
        for ck in (False, True):
            b = find_max_batch(cfg, seq, "auto", "auto", ck)
            print(f"max stable micro-batch @ seq={seq}, grad_ckpt={ck}: {b} (use a smaller value for headroom)")
        return
    has_cuda = torch.cuda.is_available()
    combos = [("fp32", "manual", False), ("fp32", "sdpa", False), ("fp32", "sdpa", True)]
    if has_cuda:
        combos += [("auto", "manual", False), ("auto", "sdpa", False), ("auto", "sdpa", True)]
    rows = []
    for prec, impl, ck in combos:
        r = bench(load_config(a.config), a.batch_size, seq, prec, impl, ck, a.iters, a.warmup)
        rows.append(r)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
