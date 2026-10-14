"""MLM evaluation: loss, masked-token top-1/top-5 accuracy (and NSP stats when enabled)."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from ..training.metrics import mlm_counts


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, dtype=torch.float32, max_batches: int | None = None) -> dict:
    was_training = model.training
    model.eval()  # dropout off
    tot_loss = tot_n = top1 = top5 = 0
    nsp_loss = nsp_correct = nsp_n = 0
    conf = torch.zeros(2, 2, dtype=torch.long)  # rows = true, cols = predicted
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device, dtype):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
        logits = out.mlm_logits.float()
        c = mlm_counts(logits, batch["labels"])
        if c["n"]:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1),
                                 ignore_index=-100, reduction="sum")
            tot_loss += float(ce); tot_n += c["n"]; top1 += c["top1"]; top5 += c["top5"]
        if out.nsp_logits is not None and "next_sentence_label" in batch:
            y = batch["next_sentence_label"]
            nl = out.nsp_logits.float()
            nsp_loss += float(F.cross_entropy(nl, y, reduction="sum"))
            pred = nl.argmax(-1)
            nsp_correct += int((pred == y).sum()); nsp_n += len(y)
            for t, p in zip(y.tolist(), pred.tolist()):
                conf[t, p] += 1
    model.train(was_training)
    res = {"mlm_loss": tot_loss / max(tot_n, 1), "mlm_acc": top1 / max(tot_n, 1),
           "mlm_top5_acc": top5 / max(tot_n, 1), "masked_tokens": tot_n}
    if nsp_n:
        res.update({"nsp_loss": nsp_loss / nsp_n, "nsp_acc": nsp_correct / nsp_n,
                    "nsp_confusion(rows=true IsNext/NotNext, cols=pred)": conf.tolist()})
    return res
