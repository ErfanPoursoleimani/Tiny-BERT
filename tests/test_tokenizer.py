def test_special_ids_fixed(tok):
    assert (tok.pad_id, tok.unk_id, tok.cls_id, tok.sep_id, tok.mask_id) == (0, 1, 2, 3, 4)


def test_encode_decode_roundtrip(tok):
    text = "the small cat sat on the old mat."
    enc = tok.encode(text)
    assert enc.input_ids[0] == tok.cls_id and enc.input_ids[-1] == tok.sep_id
    assert tok.decode(enc.input_ids) == text
    assert enc.attention_mask == [1] * len(enc.input_ids)
    assert set(enc.token_type_ids) == {0}


def test_unknown_maps_to_unk(tok):
    assert tok.unk_id in tok.encode("zzxqjk 日本語").input_ids


def test_mask_literal(tok):
    ids = tok.encode("the [MASK] sat").input_ids
    assert ids.count(tok.mask_id) == 1


def test_pair_token_types(tok):
    e = tok.encode_pair("the cat sat.", "the dog ran.")
    ids, tt = e.input_ids, e.token_type_ids
    assert ids[0] == tok.cls_id and ids.count(tok.sep_id) == 2 and ids[-1] == tok.sep_id
    first_sep = ids.index(tok.sep_id)
    assert all(t == 0 for t in tt[: first_sep + 1])
    assert all(t == 1 for t in tt[first_sep + 1:])
    assert len(ids) == len(tt) == len(e.attention_mask)


def test_pair_truncation(tok):
    e = tok.encode_pair("the cat sat on the mat. " * 20, "the dog ran. " * 20, max_length=16)
    assert len(e.input_ids) <= 16
