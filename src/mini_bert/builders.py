"""Glue code: build tokenizer, datasets, collators and model from a Config."""
from __future__ import annotations

import os

from .config import Config
from .data import PretrainingCollator, PretrainingDataset, load_processed
from .model.bert import BertForPreTraining
from .tokenizer.tokenizer import WordPieceTokenizer


def load_tokenizer(cfg: Config) -> WordPieceTokenizer:
    tok = WordPieceTokenizer(cfg.data.tokenizer_dir)
    if tok.vocab_size > cfg.model.vocab_size:
        raise ValueError(f"tokenizer vocab ({tok.vocab_size}) > model.vocab_size ({cfg.model.vocab_size})")
    return tok


def tokenizer_path(cfg: Config) -> str:
    return os.path.join(cfg.data.tokenizer_dir, "tokenizer.json")


def build_dataset(cfg: Config, tok: WordPieceTokenizer, split: str, docs=None) -> PretrainingDataset:
    docs = docs if docs is not None else load_processed(cfg.data.data_dir, split)
    return PretrainingDataset(docs, tok.cls_id, tok.sep_id, cfg.data.max_seq_length, nsp=cfg.nsp.enabled,
                              short_seq_prob=cfg.data.short_seq_prob if split == "train" else 0.0,
                              seed=cfg.training.seed)


def collator_factory(cfg: Config, tok: WordPieceTokenizer):
    def make(seed):
        return PretrainingCollator(tok.pad_id, tok.mask_id, cfg.model.vocab_size, tok.special_ids, cfg.mlm,
                                   nsp=cfg.nsp.enabled, seed=seed)
    return make


def build_model(cfg: Config) -> BertForPreTraining:
    return BertForPreTraining(cfg.model, nsp_weight=cfg.nsp.weight)
