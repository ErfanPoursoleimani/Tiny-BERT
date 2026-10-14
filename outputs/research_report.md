# Mini-BERT research report (authoring-sandbox results only)

**Read this first.** The code was written and tested in a sandbox with **1 CPU core, ~3 GB RAM, no GPU and no access
to Hugging Face**. Nothing here was run on the RTX 3050 or on WikiText-2. Every number below was measured in that
sandbox on a toy corpus; items not run are marked NOT EXECUTED.

## 1-3. Question, architecture, math
Can a conservative 6-layer/384-hidden post-LN BERT be pretrained with MLM on 6 GB VRAM? Architecture and equations: `docs/architecture.md`.

## 4-7. Data, tokenizer, objectives
* WikiText-2 / WikiText-103 / TinyStories / text files supported - **NOT EXECUTED** (no network here).
* Executed only on a *templated synthetic corpus* (360 train docs, 29.9k tokens, 153-token WordPiece vocab). It has
  almost no linguistic variety, so its metrics say nothing about real-language quality.

## 8-10. Hardware, config, parameters
Sandbox: CPU only, fp32. Smoke config (`configs/synthetic_smoke.yaml`, tiny architecture: 4 layers, 192 hidden, 4 heads): 1,907,152 parameters.
Default `configs/small.yaml`: 22,445,232 params tied / ~33.97M untied (computed; model built, not trained).

## 11-15. Measured results (CPU, synthetic corpus, 1 seed, 300 steps)
| run | val MLM loss | top-1 | top-5 | NSP acc | mean tokens/s |
|---|---|---|---|---|---|
| MLM only | 1.495 | 0.491 | 0.826 | - | 5,694 |
| MLM only, +20 steps after resume | 1.408 | 0.537 | 0.848 | - | 4,791 |
| MLM + NSP | 1.601 | 0.455 | 0.816 | 0.697 | 6,695 |

Peak VRAM: **NOT EXECUTED** (no GPU). Run `python scripts/benchmark.py` and `--find-max-batch-size` on the 3050.

Overfit test (`tests/test_training.py::test_overfit_tiny_corpus`): train MLM loss 4.63 -> 0.34, val masked acc 0.64 -> 0.94 in 150 steps.
Tests: **40 passed, 0 failed**.

## 16. Ablations
* **MLM vs MLM+NSP - run once, inconclusive.** The NSP model's confusion matrix is `[[0,40],[0,92]]`: it predicts
  NotNext for everything, so 0.697 accuracy is just the majority-class rate; NSP was **not learned** in 300 steps.
  The two runs also differ in how examples are built (pair packing vs single segment), one seed, tiny model. The
  higher MLM loss under NSP should not be read as evidence about NSP.
* Model size, masking rate, context length, vocab size, RoPE, depth-vs-width: **NOT EXECUTED** (configs exist for
  the first four; RoPE not implemented). Commands: `python scripts/train.py --config configs/experiments/<name>.yaml`.
* Downstream toy sentiment (`finetune_classifier.py --dataset toy`): pretrained and random-init both reached 1.0
  accuracy because the task is trivial, so it shows the code path works, not that pretraining helps. Frozen-encoder run got 0.51
  after 2 epochs (undertrained). SST-2 transfer experiment: **NOT EXECUTED**.

## 17. Attention
One heatmap was generated successfully for a smoke checkpoint; no interpretation is claimed.

## 18-19. Limitations / reproducibility
CPU-only, toy data, single seeds. Seeds are set for Python/NumPy/PyTorch/CUDA; CUDA is not bitwise deterministic.
Benchmark on CPU (fp32, 1.9M-param model, batch 8, seq 64): manual attention 8.0k tok/s, SDPA 8.6k, SDPA + gradient
checkpointing 5.7k - indicative only; GPU behaviour will differ.

## 20. Next experiments
1. On the 3050: `benchmark.py --find-max-batch-size`, then a 10k-step WikiText-2 run with `small.yaml`.
2. MLM vs MLM+NSP with several seeds and an equal-example-construction control (or SOP instead).
3. Fine-tune on SST-2 from the pretrained checkpoint vs random init.
