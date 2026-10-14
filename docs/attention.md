# Attention
Two implementations: `manual` (educational: scores, additive mask, softmax) and `sdpa`
(`F.scaled_dot_product_attention`, `is_causal=False`, boolean padding mask). `auto` = SDPA unless weights are requested.
A test asserts that both agree. Attention heatmaps are exploratory; do not over-interpret them.
