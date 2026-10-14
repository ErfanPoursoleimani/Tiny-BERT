import math

import pytest
import torch

from mini_bert.config import ModelConfig
from mini_bert.model.attention import MultiHeadSelfAttention


def _attn(impl, h=32, heads=4):
    return MultiHeadSelfAttention(ModelConfig(vocab_size=10, hidden_size=h, num_attention_heads=heads,
                                              intermediate_size=64, max_position_embeddings=16,
                                              attention_implementation=impl)).eval()


def test_shapes():
    a = _attn("manual")
    y, w = a(torch.randn(2, 7, 32), output_attentions=True)
    assert y.shape == (2, 7, 32) and w.shape == (2, 4, 7, 7)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 7), atol=1e-5)


def test_invalid_head_division():
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=30, num_attention_heads=4)


def test_manual_equals_sdpa():
    m, s = _attn("manual"), _attn("sdpa")
    s.load_state_dict(m.state_dict())
    x = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.long); mask[1, 6:] = 0
    ym, _ = m(x, mask); ys, _ = s(x, mask)
    assert torch.allclose(ym[0], ys[0], atol=1e-5)
    assert torch.allclose(ym[1, :6], ys[1, :6], atol=1e-5)  # pad query rows are irrelevant


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_no_causal_mask_future_affects_past(impl):
    """KEY BERT-vs-GPT test: changing token C changes the output at position B (< C)."""
    a = _attn(impl)
    x = torch.randn(1, 3, 32)
    x2 = x.clone(); x2[0, 2] += 1.0  # perturb last token only
    y1, _ = a(x); y2, _ = a(x2)
    assert not torch.allclose(y1[0, 1], y2[0, 1], atol=1e-6)  # position 1 (B) sees position 2 (C)
    assert not torch.allclose(y1[0, 0], y2[0, 0], atol=1e-6)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_padding_cannot_affect_valid_tokens(impl):
    a = _attn(impl)
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    x2 = x.clone(); x2[0, 4:] = torch.randn(2, 32) * 10
    y1, _ = a(x, mask); y2, _ = a(x2, mask)
    assert torch.allclose(y1[0, :4], y2[0, :4], atol=1e-5)
    _, w = _attn("manual")(x, mask, output_attentions=True)
    assert w[..., 4:].max() < 1e-6  # zero attention weight on pad keys
