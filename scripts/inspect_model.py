import argparse

import _common  # noqa: F401

from mini_bert.builders import build_model
from mini_bert.config import load_config
from mini_bert.utils.parameter_count import breakdown, count_params
from mini_bert.utils.profiling import estimate_memory_mb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--nsp", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.nsp:
        cfg.nsp.enabled = cfg.model.nsp_enabled = True
    m = build_model(cfg)
    mc = cfg.model
    print("=" * 44 + "\nMini-BERT\n" + "=" * 44)
    print(f"Vocabulary:       {mc.vocab_size:,}\nHidden size:      {mc.hidden_size}\nLayers:           {mc.num_hidden_layers}")
    print(f"Heads:            {mc.num_attention_heads}\nHead dimension:   {mc.head_dim}\nIntermediate:     {mc.intermediate_size}")
    print(f"Max sequence:     {mc.max_position_embeddings}\nTied embeddings:  {mc.tie_embeddings}\n\nParameters:")
    for k, v in breakdown(m).items():
        print(f"  {k + ':':<34}{v:>12,}")
    tot = count_params(m)
    print(f"\nTotal parameters:     {tot:,} (~{tot / 1e6:.1f}M)\nTrainable parameters: {count_params(m, True):,}")
    tc = cfg.training
    print("\nRough memory estimate (MB, NOT exact; fp32 weights/optimizer, bf16/fp16 activations):")
    for k, v in estimate_memory_mb(tot, batch=tc.batch_size, seq=mc.max_position_embeddings, hidden=mc.hidden_size,
                                   layers=mc.num_hidden_layers, heads=mc.num_attention_heads,
                                   inter=mc.intermediate_size).items():
        print(f"  {k:<48}{v:>9.0f}")
    print("=" * 44)


if __name__ == "__main__":
    main()
