import torch

from mini_bert.model import BertForPreTraining
from mini_bert.training.metrics import mlm_counts


def test_logits_shape_and_loss_ignores_unmasked(cfg):
    m = BertForPreTraining(cfg.model).eval()
    ids = torch.randint(5, cfg.model.vocab_size, (2, 10))
    labels = torch.full_like(ids, -100); labels[0, 2] = ids[0, 2]
    out = m(ids, labels=labels)
    assert out.mlm_logits.shape == (2, 10, cfg.model.vocab_size)
    # loss equals CE at the single labelled position
    ref = torch.nn.functional.cross_entropy(out.mlm_logits[0, 2:3], ids[0, 2:3])
    assert torch.allclose(out.loss, ref, atol=1e-5)


def test_accuracy_only_over_masked_positions():
    logits = torch.zeros(1, 4, 10); logits[0, 0, 3] = 5; logits[0, 1, 4] = 5
    labels = torch.tensor([[3, 9, -100, -100]])  # pos0 correct, pos1 wrong, others ignored
    c = mlm_counts(logits, labels)
    assert c["n"] == 2 and c["top1"] == 1
