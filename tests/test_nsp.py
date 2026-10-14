import torch

from mini_bert.data.dataset import IS_NEXT, NOT_NEXT, PretrainingDataset
from mini_bert.model import BertForPreTraining


def test_nsp_dataset_labels(tok, encoded_docs):
    ds = PretrainingDataset(encoded_docs, tok.cls_id, tok.sep_id, 32, nsp=True, seed=0)
    labels = []
    for i in range(len(ds)):
        ex = ds[i]
        ids, tt = ex["input_ids"], ex["token_type_ids"]
        assert ids[0] == tok.cls_id and ids.count(tok.sep_id) == 2 and len(ids) <= 32
        assert len(ids) == len(tt) and tt[0] == 0 and tt[-1] == 1
        labels.append(ex["next_sentence_label"])
    assert set(labels) == {IS_NEXT, NOT_NEXT}
    frac = sum(l == NOT_NEXT for l in labels) / len(labels)
    assert 0.35 < frac < 0.75


def test_nsp_is_deterministic(tok, encoded_docs):
    ds = PretrainingDataset(encoded_docs, tok.cls_id, tok.sep_id, 32, nsp=True, seed=3)
    assert ds[5] == ds[5]


def test_nsp_logits_and_loss(cfg):
    cfg.model.nsp_enabled = True
    m = BertForPreTraining(cfg.model, nsp_weight=2.0)
    ids = torch.randint(5, cfg.model.vocab_size, (4, 10))
    labels = torch.full_like(ids, -100); labels[:, 2] = ids[:, 2]
    y = torch.tensor([0, 1, 0, 1])
    out = m(ids, labels=labels, next_sentence_label=y)
    assert out.nsp_logits.shape == (4, 2)
    assert torch.allclose(out.loss, out.mlm_loss + 2.0 * out.nsp_loss)
    out.loss.backward()
    assert m.nsp_head.classifier.weight.grad is not None


def test_mlm_only_has_no_nsp_head(cfg):
    m = BertForPreTraining(cfg.model)
    assert m.nsp_head is None and m(torch.randint(5, 20, (1, 5))).nsp_logits is None
