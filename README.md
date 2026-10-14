# Mini-BERT

A from-scratch, BERT-style (encoder-only, post-LayerNorm, bidirectional) Transformer in plain PyTorch, sized for an
RTX 3050 Laptop GPU (6 GB). No pretrained weights are used anywhere.

## Quick start
```bash
pip install -r requirements.txt
pytest -q                                              # 40 tests
python scripts/inspect_model.py                        # parameter report + rough memory estimate
python scripts/prepare_data.py --config configs/small.yaml   # WikiText-2 + 30k WordPiece (needs internet once)
python scripts/benchmark.py --find-max-batch-size      # on the GPU: find a safe micro-batch
python scripts/train.py --config configs/small.yaml
python scripts/train.py --config configs/small.yaml --resume checkpoints/latest.pt
python scripts/evaluate.py --checkpoint checkpoints/best.pt
python scripts/mask_text.py --checkpoint checkpoints/best.pt --text "The capital of France is [MASK]."
python scripts/analyze_attention.py --checkpoint checkpoints/best.pt --text "The cat sat on the mat."
python scripts/finetune_classifier.py --checkpoint checkpoints/best.pt --dataset sst2
tensorboard --logdir outputs/runs
```
Offline smoke test (templated toy corpus, not a benchmark): `python scripts/prepare_data.py --config configs/synthetic_smoke.yaml && python scripts/train.py --config configs/synthetic_smoke.yaml`

## Parameter note
Default config (tied MLM decoder): **22.4M** params. With `model.tie_embeddings: false`: ~34.0M.
## Status
See `outputs/research_report.md` for what was executed in the authoring sandbox (CPU only) and what is NOT EXECUTED
(GPU benchmarks, WikiText-2 training, most ablations). Docs: `docs/`.
Not implemented: RoPE, notebooks, vocabulary-size / depth-vs-width experiment runners, UMAP/t-SNE (PCA helper only).
