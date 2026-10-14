"""Train a BERT-style WordPiece vocabulary with the `tokenizers` library."""
from __future__ import annotations

import os
from typing import Iterable

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # ids 0..4 (fixed order)


def train_wordpiece(texts: Iterable[str], vocab_size: int, out_dir: str, lowercase: bool = True,
                    min_frequency: int = 2) -> str:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    steps = [normalizers.NFD(), normalizers.StripAccents()] if lowercase else []
    if lowercase:
        steps.insert(0, normalizers.Lowercase())
    tok.normalizer = normalizers.Sequence(steps + [normalizers.NFC()]) if steps else normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix="##")
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
    )
    tok.train_from_iterator(texts, trainer)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    return path
