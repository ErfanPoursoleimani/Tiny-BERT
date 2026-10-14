"""BERT-compatible WordPiece tokenizer wrapper: encode / decode / encode_pair."""
from __future__ import annotations

import os
from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass
class Encoded:
    input_ids: list[int]
    token_type_ids: list[int]
    attention_mask: list[int]


class WordPieceTokenizer:
    PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"

    def __init__(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"tokenizer file not found: {path} (run scripts/train_tokenizer.py)")
        self.path = path
        self.tok = Tokenizer.from_file(path)
        self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id = (
            self.tok.token_to_id(t) for t in (self.PAD, self.UNK, self.CLS, self.SEP, self.MASK)
        )
        if None in (self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id):
            raise ValueError("tokenizer is missing one of [PAD] [UNK] [CLS] [SEP] [MASK]")
        self.special_ids = {self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id}

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def tokenize(self, text: str) -> list[str]:
        return self.tok.encode(text, add_special_tokens=False).tokens

    def encode_ids(self, text: str) -> list[int]:
        """Raw WordPiece ids, no special tokens."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, text: str, max_length: int | None = None, add_special_tokens: bool = True) -> Encoded:
        """text -> [CLS] ids [SEP]. Literal '[MASK]' in text maps to the mask id."""
        ids = self._ids_with_mask(text)
        if add_special_tokens:
            if max_length is not None:
                ids = ids[: max_length - 2]
            ids = [self.cls_id] + ids + [self.sep_id]
        return Encoded(ids, [0] * len(ids), [1] * len(ids))

    def encode_pair(self, a: str, b: str, max_length: int | None = None) -> Encoded:
        """[CLS] A [SEP] B [SEP]; token types: A (incl. CLS and first SEP)=0, B (incl. last SEP)=1."""
        ia, ib = self._ids_with_mask(a), self._ids_with_mask(b)
        if max_length is not None:  # truncate the longer one first
            while len(ia) + len(ib) > max_length - 3:
                (ia if len(ia) > len(ib) else ib).pop()
        ids = [self.cls_id] + ia + [self.sep_id] + ib + [self.sep_id]
        types = [0] * (len(ia) + 2) + [1] * (len(ib) + 1)
        return Encoded(ids, types, [1] * len(ids))

    def _ids_with_mask(self, text: str) -> list[int]:
        parts = text.split(self.MASK)
        ids: list[int] = []
        for i, p in enumerate(parts):
            if i > 0:
                ids.append(self.mask_id)
            if p.strip():
                ids.extend(self.encode_ids(p))
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.tok.id_to_token(i) for i in ids]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
