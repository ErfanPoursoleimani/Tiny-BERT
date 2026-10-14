import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from mini_bert.config import Config, ModelConfig  # noqa: E402
from mini_bert.data.preprocessing import synthetic_corpus  # noqa: E402
from mini_bert.tokenizer.tokenizer import WordPieceTokenizer  # noqa: E402
from mini_bert.tokenizer.train_tokenizer import train_wordpiece  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    return synthetic_corpus(n_docs=60, seed=0)


@pytest.fixture(scope="session")
def tok(tmp_path_factory, corpus):
    d = tmp_path_factory.mktemp("tok")
    train_wordpiece((s for doc in corpus for s in doc), 300, str(d))
    return WordPieceTokenizer(str(d))


@pytest.fixture()
def cfg(tok):
    c = Config()
    c.model = ModelConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=64, max_position_embeddings=32)
    c.data.max_seq_length = 32
    c.__post_init__()
    return c


@pytest.fixture()
def encoded_docs(tok, corpus):
    return [[tok.encode_ids(s) for s in d] for d in corpus]
