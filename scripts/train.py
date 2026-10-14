import argparse

import _common  # noqa: F401

from mini_bert.builders import build_dataset, build_model, collator_factory, load_tokenizer, tokenizer_path
from mini_bert.config import load_config
from mini_bert.training.trainer import Trainer
from mini_bert.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--nsp", action="store_true", help="enable NSP")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    a = ap.parse_args()
    ov = {"training": {}}
    if a.seed is not None: ov["training"]["seed"] = a.seed
    if a.max_steps is not None: ov["training"]["max_steps"] = a.max_steps
    if a.batch_size is not None: ov["training"]["batch_size"] = a.batch_size
    if a.grad_accum is not None: ov["training"]["gradient_accumulation_steps"] = a.grad_accum
    if a.nsp: ov["nsp"] = {"enabled": True}
    cfg = load_config(a.config, ov)
    set_seed(cfg.training.seed)
    tok = load_tokenizer(cfg)
    train_ds, val_ds = build_dataset(cfg, tok, "train"), build_dataset(cfg, tok, "validation")
    tr = Trainer(cfg, build_model(cfg), train_ds, val_ds, collator_factory(cfg, tok), tokenizer_path(cfg))
    s = tr.train(resume=a.resume)
    print("run dir:", tr.run_dir)
    print(s)


if __name__ == "__main__":
    main()
