"""Masked-token prediction. Multiple masks are predicted independently (each conditioned on the
unmasked context only), which is an approximation: BERT's MLM was not trained autoregressively."""
from __future__ import annotations

import torch


@torch.no_grad()
def predict_masks(model, tok, text: str, top_k: int = 5, temperature: float = 1.0, device="cpu") -> list[dict]:
    model.eval()
    enc = tok.encode(text, max_length=model.cfg.max_position_embeddings)
    ids = torch.tensor([enc.input_ids], device=device)
    pos = [i for i, t in enumerate(enc.input_ids) if t == tok.mask_id]
    if not pos:
        raise ValueError("text contains no [MASK] token")
    logits = model(ids, torch.tensor([enc.token_type_ids], device=device),
                   torch.tensor([enc.attention_mask], device=device)).mlm_logits[0]
    results = []
    for p in pos:
        lg = logits[p].float() / max(temperature, 1e-6)
        lg[list(tok.special_ids)] = float("-inf")  # never predict special tokens
        probs = torch.softmax(lg, -1)
        top = probs.topk(top_k)
        results.append({"position": p, "predictions": [
            (tok.convert_ids_to_tokens([i])[0], float(pr)) for pr, i in zip(top.values.tolist(), top.indices.tolist())]})
    return results
