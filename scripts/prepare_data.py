"""Download/load corpus, train WordPiece tokenizer on the TRAIN split only, encode, save token ids."""
import argparse
import os

import _common  # noqa: F401
import torch

from mini_bert.config import load_config
from mini_bert.data.preprocessing import load_corpus
from mini_bert.tokenizer.train_tokenizer import train_wordpiece
from mini_bert.tokenizer.tokenizer import WordPieceTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--dataset", default=None, help="override data.dataset")
    ap.add_argument("--vocab-size", type=int, default=None, help="override model.vocab_size for the tokenizer")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ds = a.dataset or cfg.data.dataset
    vocab = a.vocab_size or cfg.model.vocab_size
    corpus = load_corpus(ds, cfg.data.text_path, seed=a.seed)
    print({k: f"{len(v)} docs / {sum(len(d) for d in v)} sentences" for k, v in corpus.items()})
    tok_dir = cfg.data.tokenizer_dir
    if not os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        # tokenizer sees training text only -> no leakage into validation/test
        train_wordpiece((s for d in corpus["train"] for s in d), vocab, tok_dir)
    tok = WordPieceTokenizer(tok_dir)
    print(f"tokenizer vocab size: {tok.vocab_size}")
    os.makedirs(cfg.data.data_dir, exist_ok=True)
    for split, docs in corpus.items():
        enc = [[tok.encode_ids(s) for s in d] for d in docs]
        enc = [[s for s in d if s] for d in enc]
        torch.save({"docs": enc, "dataset": ds, "tokenizer": tok.path}, os.path.join(cfg.data.data_dir, f"{split}.pt"))
        print(f"saved {split}: {len(enc)} docs, {sum(len(s) for d in enc for s in d)} tokens")


if __name__ == "__main__":
    main()
