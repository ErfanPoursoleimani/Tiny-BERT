"""Downstream classification with the [CLS] representation: pretrained-frozen / pretrained-finetuned / random-init.

Dataset: --dataset sst2 (GLUE via `datasets`, needs network) or --dataset toy (offline synthetic sanity task).
Reports accuracy, precision, recall, F1 and macro-F1.
"""
import argparse
import random

import _common  # noqa: F401
import torch
from torch.utils.data import DataLoader

from mini_bert.builders import load_tokenizer
from mini_bert.config import Config
from mini_bert.model.bert import BertForSequenceClassification
from mini_bert.training.optimizer import build_optimizer
from mini_bert.utils.device import select_device_and_precision
from mini_bert.utils.seed import set_seed


def toy_data(n, seed):
    rng = random.Random(seed)
    pos = ["happy", "quiet", "small", "quick"]; neg = ["old", "brown"]
    rows = []
    for _ in range(n):
        y = rng.random() < 0.5
        w = rng.choice(pos if y else neg)
        rows.append((f"The {w} cat sat on the mat.", int(y)))
    return rows


def load_rows(name, split, n):
    if name == "toy":
        return toy_data(n, 0 if split == "train" else 1)
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="train" if split == "train" else "validation")
    return [(r["sentence"], r["label"]) for r in ds.select(range(min(n, len(ds))))]


def metrics(y, p):
    import numpy as np
    y, p = np.array(y), np.array(p)
    out = {"accuracy": float((y == p).mean())}
    f1s = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-12)
        out[f"class{c}"] = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}; f1s.append(f1)
    out["macro_f1"] = float(sum(f1s) / 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="omit for random init")
    ap.add_argument("--config-from-checkpoint", default=None)
    ap.add_argument("--dataset", default="toy", choices=["toy", "sst2"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    src = a.checkpoint or a.config_from_checkpoint
    if not src:
        raise SystemExit("need --checkpoint (pretrained) or --config-from-checkpoint (random init, same arch)")
    st = torch.load(src, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(st["config"])
    tok = load_tokenizer(cfg)
    prec = select_device_and_precision("auto"); dev = prec.device
    model = BertForSequenceClassification(cfg.model, 2, a.freeze)
    if a.checkpoint:
        model.load_pretrained_encoder(st["model"])
    model.to(dev)

    def encode(rows):
        out = []
        for t, y in rows:
            e = tok.encode(t, cfg.model.max_position_embeddings); out.append((e.input_ids, e.token_type_ids, y))
        return out

    def collate(b):
        T = max(len(x[0]) for x in b)
        ids = torch.zeros(len(b), T, dtype=torch.long); tt = torch.zeros_like(ids); am = torch.zeros_like(ids)
        for i, (x, t, _) in enumerate(b):
            ids[i, :len(x)] = torch.tensor(x); tt[i, :len(x)] = torch.tensor(t); am[i, :len(x)] = 1
        return ids, tt, am, torch.tensor([x[2] for x in b])

    tr = DataLoader(encode(load_rows(a.dataset, "train", a.n_train)), 32, shuffle=True, collate_fn=collate)
    va = DataLoader(encode(load_rows(a.dataset, "val", a.n_val)), 64, collate_fn=collate)
    opt = build_optimizer(model, a.lr, 0.01)
    for ep in range(a.epochs):
        model.train()
        for ids, tt, am, y in tr:
            loss, _ = model(ids.to(dev), tt.to(dev), am.to(dev), y.to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for ids, tt, am, y in va:
                _, lg = model(ids.to(dev), tt.to(dev), am.to(dev)); ps += lg.argmax(-1).tolist(); ys += y.tolist()
        print(f"epoch {ep + 1}:", metrics(ys, ps))


if __name__ == "__main__":
    main()
