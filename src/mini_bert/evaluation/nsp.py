"""NSP helpers. Label convention: 0 = IsNext, 1 = NotNext."""
from __future__ import annotations


def format_confusion(conf: list[list[int]]) -> str:
    return ("            pred:IsNext  pred:NotNext\n"
            f"true:IsNext   {conf[0][0]:>9}  {conf[0][1]:>11}\n"
            f"true:NotNext  {conf[1][0]:>9}  {conf[1][1]:>11}")
