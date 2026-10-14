# Training
AdamW (no decay on bias/LayerNorm), linear warmup + linear/cosine decay, grad clip 1.0, gradient accumulation,
BF16 > FP16(+GradScaler) > FP32 auto-selection, optional gradient checkpointing, checkpoints (`latest/best/step_N`)
with model/optimizer/scheduler/scaler/RNG/config/tokenizer hash. Resume: `--resume checkpoints/latest.pt`.
Seeds: Python/NumPy/PyTorch/CUDA; bitwise determinism on CUDA is not guaranteed.
Each run writes `outputs/runs/<timestamp>_<name>/{config.yaml,metrics.json,training.log,plots/,tensorboard/}`.
Note: `checkpoints/` (not the run dir) holds checkpoints unless a run_dir is passed to `Trainer`.
