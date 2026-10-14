# Masked language modeling
~15% of eligible tokens are selected per batch, freshly each time a batch is collated (dynamic masking; seedable).
Of those: 80% -> `[MASK]`, 10% -> random non-special token, 10% unchanged. Labels are the original ids at selected
positions and -100 elsewhere. `[PAD] [UNK] [CLS] [SEP] [MASK]` are never selected. If a sequence would get zero
masks, one eligible token is masked so the loss is defined.
Metrics are computed on masked positions only (loss, top-1, top-5). Pseudo-perplexity (`evaluation/perplexity.py`)
masks one token at a time and exponentiates the mean NLL; it is NOT comparable to GPT perplexity.
## NSP caveat
Original BERT used NSP; RoBERTa and others found it unhelpful or harmful, and ALBERT proposed sentence-order prediction.
NSP is off by default and meant to be studied empirically (`configs/experiments/mlm_nsp.yaml`).
