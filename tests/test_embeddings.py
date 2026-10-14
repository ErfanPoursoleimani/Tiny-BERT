import torch

from mini_bert.model.embeddings import BertEmbeddings


def test_shape_and_sum(cfg):
    emb = BertEmbeddings(cfg.model).eval()
    ids = torch.randint(5, cfg.model.vocab_size, (3, 10))
    out = emb(ids)
    assert out.shape == (3, 10, cfg.model.hidden_size)
    # LayerNorm output is ~zero-mean per position
    assert out.mean(-1).abs().max() < 1e-4


def test_token_type_changes_output(cfg):
    emb = BertEmbeddings(cfg.model).eval()
    ids = torch.randint(5, cfg.model.vocab_size, (1, 8))
    a = emb(ids, torch.zeros_like(ids))
    b = emb(ids, torch.ones_like(ids))
    assert not torch.allclose(a, b)


def test_position_matters(cfg):
    emb = BertEmbeddings(cfg.model).eval()
    ids = torch.full((1, 6), 7)
    out = emb(ids)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_too_long_raises(cfg):
    import pytest
    emb = BertEmbeddings(cfg.model)
    with pytest.raises(ValueError):
        emb(torch.zeros(1, cfg.model.max_position_embeddings + 1, dtype=torch.long))
