"""File + console logging and optional TensorBoard writer."""
from __future__ import annotations

import logging
import os


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def get_tb_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir)
    except Exception:  # tensorboard not installed
        return None
