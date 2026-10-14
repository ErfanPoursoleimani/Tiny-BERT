# Architecture
Embeddings: `E = LN(E_tok + E_pos + E_seg)` then dropout (learned positions, `nn.Embedding(max_len, H)`).
Layer (post-LN): `x -> LN(x + Drop(MHSA(x))) -> LN(. + Drop(FFN(.)))`, `FFN(x) = W2 GELU(W1 x + b1) + b2` (384 -> 1536 -> 384).
MHSA: `Q=XW_Q, K=XW_K, V=XW_V`; heads split to `(B,h,T,d)` with `d=H/h=64`; `A=softmax(QK^T/sqrt(d)+M)`; `Z=AV`; `Concat(heads) W_O`.
LayerNorm: `gamma (x-mu)/sqrt(sigma^2+eps) + beta`, `eps=1e-12`.
MLM head: `Linear -> GELU -> LayerNorm -> decoder(V)`; decoder tied to the token embedding by default.
NSP head (optional): `H[:,0] -> Linear -> Tanh (pooler) -> Linear(H,2)`. Label 0 = IsNext, 1 = NotNext.
Loss: `L_MLM = -sum_{i in M} log P(x_i | x_{not M})`; with NSP `L = L_MLM + lambda L_NSP` (default lambda = 1).

## Parameter counts (measured with `scripts/inspect_model.py`)
* default config, tied output embeddings: **22,445,232** (~22.4M)
* same config with `tie_embeddings: false`: ~34.0M (adds a second 30000x384 matrix = 11.52M)
So the "25-35M" target is met only untied; tying is a deliberate memory-saving default.
