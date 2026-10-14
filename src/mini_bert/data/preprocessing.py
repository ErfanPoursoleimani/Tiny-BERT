"""Corpus loading and sentence splitting. Each split is a list of documents; a document is a list of sentences."""

from __future__ import annotations

import os
import random
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(paragraph: str) -> list[str]:
    return [
        s.strip() for s in _SENT_SPLIT.split(paragraph.strip()) if len(s.strip()) > 1
    ]


def lines_to_documents(
    lines: list[str], wikitext_headings: bool = True
) -> list[list[str]]:
    """Group lines into documents. WikiText level-1 headings ' = Title = ' start a new document."""
    docs: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        is_h1 = wikitext_headings and re.fullmatch(r"=\s[^=].*[^=]\s=", s) is not None
        if is_h1:
            if cur:
                docs.append(cur)
            cur = []
            continue
        if s.startswith("="):  # sub-heading
            continue
        cur.extend(
            split_sentences(
                s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
            )
        )
    if cur:
        docs.append(cur)
    return [d for d in docs if len(d) >= 2]


def synthetic_corpus(n_docs: int = 400, seed: int = 0) -> list[list[str]]:
    """Small templated corpus (offline testing only; NOT a real language benchmark)."""
    rng = random.Random(seed)
    animals = ["cat", "dog", "bird", "horse", "fox", "rabbit", "mouse", "owl"]
    places = ["mat", "park", "house", "barn", "river", "garden", "forest", "hill"]
    verbs = [
        "sat on",
        "ran to",
        "flew over",
        "slept near",
        "jumped over",
        "walked around",
    ]
    adjs = ["small", "old", "quick", "quiet", "brown", "happy"]
    docs = []
    for _ in range(n_docs):
        a, p = rng.choice(animals), rng.choice(places)
        doc = []
        for _ in range(rng.randint(4, 9)):
            doc.append(
                f"The {rng.choice(adjs)} {a} {rng.choice(verbs)} the {rng.choice(adjs)} {p}."
            )
            if rng.random() < 0.4:
                doc.append(f"Then the {a} saw the {rng.choice(places)} and went home.")
        docs.append(doc)
    return docs


def load_corpus(
    dataset: str, text_path: str = "", seed: int = 0
) -> dict[str, list[list[str]]]:
    """Return {'train','validation','test'} -> documents. Uses official splits when they exist."""
    if dataset == "synthetic":
        docs = synthetic_corpus(seed=seed)
        return _ratio_split(docs, seed)
    if dataset in ("wikitext-2", "wikitext-103"):
        from datasets import load_dataset  # network needed on first use

        name = {
            "wikitext-2": "wikitext-2-raw-v1",
            "wikitext-103": "wikitext-103-raw-v1",
        }[dataset]
        ds = load_dataset("Salesforce/wikitext", name)
        return {
            k: lines_to_documents(ds[k]["text"])
            for k in ("train", "validation", "test")
        }
    if dataset == "tinystories":
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train[:50000]")
        docs = [split_sentences(t) for t in ds["text"]]
        return _ratio_split([d for d in docs if len(d) >= 2], seed)
    if dataset == "text_files":
        paths = []
        if os.path.isdir(text_path):
            paths = [
                os.path.join(text_path, f)
                for f in sorted(os.listdir(text_path))
                if f.endswith(".txt")
            ]
        elif os.path.isfile(text_path):
            paths = [text_path]
        if not paths:
            raise FileNotFoundError(f"no .txt files at data.text_path={text_path!r}")
        docs = []
        for p in paths:  # blank-line separated paragraphs = documents
            with open(p, encoding="utf-8") as f:
                for para in f.read().split("\n\n"):
                    sents = split_sentences(para.replace("\n", " "))
                    if len(sents) >= 2:
                        docs.append(sents)
        return _ratio_split(docs, seed)
    raise ValueError(f"unknown dataset {dataset!r}")


def _ratio_split(docs: list[list[str]], seed: int) -> dict[str, list[list[str]]]:
    from .splits import split_documents

    return split_documents(docs, seed=seed)
