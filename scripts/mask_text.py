import argparse

import _common  # noqa: F401
import torch

from mini_bert.builders import build_model, load_tokenizer
from mini_bert.config import Config
from mini_bert.evaluation.predict import predict_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    st = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(st["config"])
    model = build_model(cfg); model.load_state_dict(st["model"]); model.eval()
    tok = load_tokenizer(cfg)
    print(a.text)
    for r in predict_masks(model, tok, a.text, a.top_k, a.temperature):
        print(f"\n[MASK] at token position {r['position']}:")
        for i, (t, p) in enumerate(r["predictions"], 1):
            print(f"  {i}. {t}  ({p:.3f})")
    if a.text.count("[MASK]") > 1:
        print("\nNote: multiple masks are predicted independently (an approximation).")


if __name__ == "__main__":
    main()
