import pytest
import torch

from mini_bert.config import MLMConfig
from mini_bert.data.collator import PretrainingCollator
from mini_bert.data.masking import mask_tokens

SPECIAL = {0, 1, 2, 3, 4}


def _batch(n=64, t=40, vocab=300):
    ids = torch.randint(5, vocab, (n, t))
    ids[:, 0] = 2; ids[:, -1] = 3; ids[:, -8:-1] = 0  # CLS, SEP, padding
    return ids


def test_specials_never_masked_and_labels():
    ids = _batch()
    out, labels = mask_tokens(ids, SPECIAL, 4, 300, MLMConfig(), torch.Generator().manual_seed(0))
    sel = labels != -100
    assert not sel[:, 0].any() and not sel[:, -1].any() and not sel[:, -8:-1].any()
    assert (labels[sel] == ids[sel]).all()          # labels hold the ORIGINAL id
    assert (out[~sel] == ids[~sel]).all()           # non-selected tokens untouched
    assert (out[:, -8:-1] == 0).all()               # padding intact


def test_rates():
    ids = _batch(n=400)
    cfg = MLMConfig()
    out, labels = mask_tokens(ids, SPECIAL, 4, 300, cfg, torch.Generator().manual_seed(1))
    sel = labels != -100
    eligible = (ids >= 5).sum()
    assert abs(sel.sum() / eligible - 0.15) < 0.01
    frac_mask = (out[sel] == 4).float().mean()
    frac_same = (out[sel] == ids[sel]).float().mean()
    assert abs(frac_mask - 0.8) < 0.02
    assert abs(frac_same - 0.1) < 0.03  # 10% unchanged (+ rare random collisions)
    rnd = sel & (out != 4) & (out != ids)
    assert (out[rnd] >= 5).all()                    # random tokens are never special


def test_dynamic_and_deterministic():
    ids = _batch(n=8)
    a = mask_tokens(ids, SPECIAL, 4, 300, MLMConfig(), torch.Generator().manual_seed(0))[1]
    b = mask_tokens(ids, SPECIAL, 4, 300, MLMConfig(), torch.Generator().manual_seed(0))[1]
    c = mask_tokens(ids, SPECIAL, 4, 300, MLMConfig(), torch.Generator().manual_seed(1))[1]
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_unk_not_masked():
    ids = torch.full((4, 20), 1); ids[:, 0] = 2
    _, labels = mask_tokens(ids, SPECIAL, 4, 300, MLMConfig(probability=0.9))
    assert (labels == -100).all()


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        MLMConfig(mask_probability=0.8, random_probability=0.2, unchanged_probability=0.2)


def test_collator_pads_and_masks(tok):
    col = PretrainingCollator(tok.pad_id, tok.mask_id, tok.vocab_size, tok.special_ids, MLMConfig())
    b = col([{"input_ids": [2, 10, 11, 12, 3], "token_type_ids": [0] * 5, "next_sentence_label": -1},
             {"input_ids": [2, 10, 3], "token_type_ids": [0] * 3, "next_sentence_label": -1}])
    assert b["input_ids"].shape == (2, 5) and b["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert (b["labels"][1, 3:] == -100).all()
