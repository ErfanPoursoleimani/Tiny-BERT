# BERT vs GPT

| | BERT (this repo) | GPT |
|---|---|---|
| Stack | encoder-only, **post-LayerNorm** | decoder-only, usually pre-LayerNorm (GPT-2) |
| Attention | **bidirectional** (only a padding mask) | **causal** (lower-triangular mask) |
| Objective | masked language modeling (+ optional NSP) | next-token prediction |
| Output use | contextual representations / classification | autoregressive text generation |

In `attention.py` there is no triangular term: `A = softmax(QK^T/sqrt(d) + M)` where `M` only masks `[PAD]` keys.
`tests/test_attention.py::test_no_causal_mask_future_affects_past` and `tests/test_encoder.py::test_bidirectional_representation`
prove that perturbing a *later* token changes the representation of an *earlier* one - impossible in a causal model.
Multi-`[MASK]` "generation" is an independent-position approximation, not autoregressive sampling.
