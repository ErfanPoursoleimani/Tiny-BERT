# Tokenizer
WordPiece via the `tokenizers` library (lowercase + accent strip, BERT pre-tokenizer, `##` continuation).
Special ids are fixed: `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`. The vocabulary is trained on the **train split only**.
`encode_pair` yields `[CLS] A [SEP] B [SEP]` with token types A=0 (incl. CLS and first SEP), B=1 (incl. last SEP).
A literal `[MASK]` in input text maps to the mask id.
