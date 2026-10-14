import argparse
import os

import _common  # noqa: F401
import torch

from mini_bert.analysis.attention_maps import plot_head
from mini_bert.builders import build_model, load_tokenizer
from mini_bert.config import Config
from mini_bert.evaluation.attention import get_attention


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--layer", type=int, default=None, help="1-based; default all")
    ap.add_argument("--head", type=int, default=None, help="1-based; default all")
    ap.add_argument("--out-dir", default="outputs/attention")
    a = ap.parse_args()
    st = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(st["config"])
    model = build_model(cfg); model.load_state_dict(st["model"]); model.eval()
    tok = load_tokenizer(cfg)
    enc = tok.encode(a.text, cfg.model.max_position_embeddings)
    attn = get_attention(model, torch.tensor(enc.input_ids))
    toks = tok.convert_ids_to_tokens(enc.input_ids)
    L, H = attn.shape[:2]
    for l in ([a.layer - 1] if a.layer else range(L)):
        for h in ([a.head - 1] if a.head else range(H)):
            plot_head(attn, toks, l, h, os.path.join(a.out_dir, f"layer_{l + 1}_head_{h + 1}.png"))
    print(f"saved {len([f for f in os.listdir(a.out_dir)])} heatmaps to {a.out_dir}")
    print("Note: attention maps are exploratory; they do not prove a head performs a linguistic function.")


if __name__ == "__main__":
    main()
