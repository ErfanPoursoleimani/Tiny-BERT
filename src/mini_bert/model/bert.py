"""BertModel (encoder), BertForPreTraining (MLM [+ NSP]) and BertForSequenceClassification."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from .embeddings import BertEmbeddings
from .encoder import BertEncoder
from .heads import MLMHead, NSPHead, Pooler


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, std)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class BertModel(nn.Module):
    """Returns last hidden states (B, T, H)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self.apply(lambda m: init_weights(m, cfg.initializer_range))

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.encoder.gradient_checkpointing = enabled

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_attentions=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)
        return self.encoder(x, attention_mask, output_attentions)


@dataclass
class PretrainingOutput:
    loss: torch.Tensor | None
    mlm_loss: torch.Tensor | None
    nsp_loss: torch.Tensor | None
    mlm_logits: torch.Tensor
    nsp_logits: torch.Tensor | None
    hidden_states: torch.Tensor
    attentions: list | None = None


class BertForPreTraining(nn.Module):
    def __init__(self, cfg: ModelConfig, nsp_weight: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.nsp_weight = nsp_weight
        self.bert = BertModel(cfg)
        self.mlm_head = MLMHead(cfg, self.bert.embeddings.token_embeddings.weight)
        self.pooler = Pooler(cfg) if cfg.nsp_enabled else None
        self.nsp_head = NSPHead(cfg) if cfg.nsp_enabled else None
        # re-init heads (tied decoder weight is re-initialised identically; harmless)
        for m in (self.mlm_head, self.pooler, self.nsp_head):
            if m is not None:
                m.apply(lambda mod: init_weights(mod, cfg.initializer_range))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                next_sentence_label=None, output_attentions=False) -> PretrainingOutput:
        h, attn = self.bert(input_ids, token_type_ids, attention_mask, output_attentions)
        mlm_logits = self.mlm_head(h)
        nsp_logits = self.nsp_head(self.pooler(h)) if self.nsp_head is not None else None
        mlm_loss = nsp_loss = loss = None
        if labels is not None:
            # float32 for numerical stability under autocast
            mlm_loss = F.cross_entropy(
                mlm_logits.float().view(-1, mlm_logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            loss = mlm_loss
        if nsp_logits is not None and next_sentence_label is not None:
            nsp_loss = F.cross_entropy(nsp_logits.float(), next_sentence_label)
            loss = nsp_loss * self.nsp_weight if loss is None else loss + self.nsp_weight * nsp_loss
        return PretrainingOutput(loss, mlm_loss, nsp_loss, mlm_logits, nsp_logits, h, attn)


class BertForSequenceClassification(nn.Module):
    """[CLS] -> Dropout -> Linear -> logits. Supports frozen or fine-tuned encoder."""

    def __init__(self, cfg: ModelConfig, num_labels: int = 2, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.bert = BertModel(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)
        init_weights(self.classifier, cfg.initializer_range)
        self.set_encoder_frozen(freeze_encoder)

    def set_encoder_frozen(self, frozen: bool) -> None:
        for p in self.bert.parameters():
            p.requires_grad = not frozen

    def load_pretrained_encoder(self, state_dict: dict) -> None:
        """Load `bert.*` weights from a BertForPreTraining checkpoint."""
        sd = {k[len("bert."):]: v for k, v in state_dict.items() if k.startswith("bert.")}
        self.bert.load_state_dict(sd)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        h, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(self.dropout(h[:, 0]))
        loss = F.cross_entropy(logits.float(), labels) if labels is not None else None
        return loss, logits
