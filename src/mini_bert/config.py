"""Dataclass configuration system with validation and YAML loading."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 30000
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 256
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    tie_embeddings: bool = True
    attention_implementation: str = "auto"  # auto | manual | sdpa
    position_embedding_type: str = "learned"  # learned | rope (rope: NOT IMPLEMENTED)
    nsp_enabled: bool = False
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.attention_implementation not in ("auto", "manual", "sdpa"):
            raise ValueError("attention_implementation must be auto|manual|sdpa")
        if self.hidden_act != "gelu":
            raise ValueError("only hidden_act='gelu' is supported")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class MLMConfig:
    probability: float = 0.15
    mask_probability: float = 0.80
    random_probability: float = 0.10
    unchanged_probability: float = 0.10

    def __post_init__(self) -> None:
        s = self.mask_probability + self.random_probability + self.unchanged_probability
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"mask/random/unchanged probabilities must sum to 1, got {s}")
        if not 0.0 < self.probability < 1.0:
            raise ValueError("mlm.probability must be in (0, 1)")


@dataclass
class NSPConfig:
    enabled: bool = False
    weight: float = 1.0  # lambda in L = L_MLM + lambda * L_NSP


@dataclass
class DataConfig:
    dataset: str = "wikitext-2"  # wikitext-2 | wikitext-103 | tinystories | text_files | synthetic
    text_path: str = ""  # for text_files
    max_seq_length: int = 256
    data_dir: str = "data/processed"
    tokenizer_dir: str = "data/tokenizer"
    short_seq_prob: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 10000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    scheduler: str = "linear"  # linear | cosine
    eval_every: int = 500
    save_every: int = 500
    eval_batches: int = 50
    log_every: int = 50
    gradient_checkpointing: bool = False
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    seed: int = 42
    num_workers: int = 0
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "baseline"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mlm: MLMConfig = field(default_factory=MLMConfig)
    nsp: NSPConfig = field(default_factory=NSPConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        self.model.nsp_enabled = self.nsp.enabled
        if self.data.max_seq_length > self.model.max_position_embeddings:
            raise ValueError("data.max_seq_length exceeds model.max_position_embeddings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        sub = {"model": ModelConfig, "mlm": MLMConfig, "nsp": NSPConfig,
               "data": DataConfig, "training": TrainingConfig}
        kwargs = {}
        for name, klass in sub.items():
            raw = d.get(name, {}) or {}
            valid = {f.name for f in fields(klass)}
            unknown = set(raw) - valid
            if unknown:
                raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}")
            kwargs[name] = klass(**raw)
        return cls(**kwargs)


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: dict | None = None) -> Config:
    """Load YAML. A top-level `base:` key names another YAML to inherit from."""
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("base", None)
    if base:
        import os
        bpath = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        with open(bpath) as f:
            d = _deep_update(yaml.safe_load(f) or {}, d)
    if overrides:
        d = _deep_update(d, overrides)
    return Config.from_dict(d)
