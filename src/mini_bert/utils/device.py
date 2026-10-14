"""Device and precision selection (prefers BF16, then FP16 + GradScaler, then FP32)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrecisionInfo:
    device: torch.device
    dtype: torch.dtype
    name: str
    use_scaler: bool
    bf16_supported: bool
    fp16_supported: bool


def select_device_and_precision(requested: str = "auto") -> PrecisionInfo:
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    fp16 = cuda
    if requested == "auto":
        name = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
    else:
        name = requested
        if name == "bf16" and not bf16:
            raise RuntimeError("bf16 requested but not supported on this device")
        if name == "fp16" and not fp16:
            raise RuntimeError("fp16 requested but CUDA is unavailable")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
    return PrecisionInfo(device, dtype, name, name == "fp16", bf16, fp16)


def print_hardware_banner(info: PrecisionInfo) -> None:
    print("=" * 44)
    print("Mini-BERT")
    print("=" * 44)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}\nVRAM: {p.total_memory / 2**30:.1f} GiB\nCUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"BF16: {info.bf16_supported}  FP16: {info.fp16_supported}")
    print(f"Selected device: {info.device}\nSelected precision: {info.name}")
