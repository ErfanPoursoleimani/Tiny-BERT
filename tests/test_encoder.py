import torch

from mini_bert.model import BertForPreTraining, BertModel


def test_forward_shape_and_backward(cfg):
    m = BertModel(cfg.model)
    ids = torch.randint(5, cfg.model.vocab_size, (3, 12))
    h, _ = m(ids)
    assert h.shape == (3, 12, cfg.model.hidden_size)
    h.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_post_ln_structure(cfg):
    layer = BertModel(cfg.model).encoder.layers[0]
    assert hasattr(layer, "attn_norm") and hasattr(layer, "ffn_norm")  # post-LN (original BERT)
    assert type(layer.ffn.act).__name__ == "GELU"


def test_bidirectional_representation(cfg):
    """Changing token C changes the representation of B (non-causal), and padding does not leak."""
    m = BertModel(cfg.model).eval()
    ids = torch.tensor([[5, 6, 7, 0]]); mask = torch.tensor([[1, 1, 1, 0]])
    ids2 = ids.clone(); ids2[0, 2] = 9           # change C
    h1, _ = m(ids, attention_mask=mask); h2, _ = m(ids2, attention_mask=mask)
    assert not torch.allclose(h1[0, 1], h2[0, 1], atol=1e-6)
    ids3 = ids.clone(); ids3[0, 3] = 11          # change PAD token id
    h3, _ = m(ids3, attention_mask=mask)
    assert torch.allclose(h1[0, :3], h3[0, :3], atol=1e-5)


def test_dropout_off_in_eval(cfg):
    m = BertModel(cfg.model).eval()
    ids = torch.randint(5, cfg.model.vocab_size, (2, 8))
    assert torch.allclose(m(ids)[0], m(ids)[0])


def test_tied_weights(cfg):
    m = BertForPreTraining(cfg.model)
    assert m.mlm_head.decoder.weight is m.bert.embeddings.token_embeddings.weight
    cfg.model.tie_embeddings = False
    m2 = BertForPreTraining(cfg.model)
    assert m2.mlm_head.decoder.weight is not m2.bert.embeddings.token_embeddings.weight


def test_gradient_checkpointing_matches(cfg):
    """With dropout disabled (via config, so SDPA's dropout_p is also 0) checkpointing must not change loss/grads."""
    cfg.model.hidden_dropout_prob = 0.0
    cfg.model.attention_probs_dropout_prob = 0.0
    m = BertForPreTraining(cfg.model).train()
    ids = torch.randint(5, cfg.model.vocab_size, (2, 8)); labels = torch.full_like(ids, -100); labels[:, 3] = 7
    m.bert.set_gradient_checkpointing(False)
    l1 = m(ids, labels=labels).loss; l1.backward()
    g1 = m.bert.embeddings.token_embeddings.weight.grad.clone(); m.zero_grad()
    m.bert.set_gradient_checkpointing(True)
    l2 = m(ids, labels=labels).loss; l2.backward()
    assert torch.allclose(l1, l2, atol=1e-6)
    assert torch.allclose(g1, m.bert.embeddings.token_embeddings.weight.grad, atol=1e-6)
