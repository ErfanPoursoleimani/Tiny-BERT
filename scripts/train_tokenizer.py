"""Train only the tokenizer (prepare_data.py does this automatically if missing)."""
import argparse

import _common  # noqa: F401

from mini_bert.config import load_config
from mini_bert.data.preprocessing import load_corpus
from mini_bert.tokenizer.train_tokenizer import train_wordpiece


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    corpus = load_corpus(cfg.data.dataset, cfg.data.text_path)
    path = train_wordpiece((s for d in corpus["train"] for s in d), a.vocab_size or cfg.model.vocab_size,
                           a.out_dir or cfg.data.tokenizer_dir)
    print("saved", path)


if __name__ == "__main__":
    main()
