import argparse

import _common  # noqa: F401
import torch
from torch.utils.data import DataLoader

from mini_bert.builders import build_dataset, build_model, collator_factory, load_tokenizer
from mini_bert.config import Config
from mini_bert.evaluation.mlm import evaluate
from mini_bert.evaluation.nsp import format_confusion
from mini_bert.utils.device import select_device_and_precision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--max-batches", type=int, default=None)
    a = ap.parse_args()
    st = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(st["config"])
    prec = select_device_and_precision("auto")
    model = build_model(cfg); model.load_state_dict(st["model"]); model.to(prec.device)
    tok = load_tokenizer(cfg)
    ds = build_dataset(cfg, tok, a.split)
    dl = DataLoader(ds, batch_size=cfg.training.batch_size, collate_fn=collator_factory(cfg, tok)(1234))
    res = evaluate(model, dl, prec.device, prec.dtype, a.max_batches)
    for k, v in res.items():
        if "confusion" in k:
            print(format_confusion(v))
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
