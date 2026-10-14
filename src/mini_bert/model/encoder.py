"""Stack of encoder layers with optional gradient checkpointing."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .encoder_layer import BertEncoderLayer


class BertEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(BertEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.gradient_checkpointing = False

    def forward(self, x, attention_mask=None, output_attentions: bool = False):
        all_attn = [] if output_attentions else None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not output_attentions:
                x, w = checkpoint(layer, x, attention_mask, False, use_reentrant=False)
            else:
                x, w = layer(x, attention_mask, output_attentions)
            if output_attentions:
                all_attn.append(w)
        return x, all_attn
