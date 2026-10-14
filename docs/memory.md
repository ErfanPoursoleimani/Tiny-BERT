# Memory
Training memory ~ weights + gradients + AdamW state (two extra fp32 tensors per parameter, i.e. 2x the fp32 weights)
+ activations + temporaries. `utils/profiling.estimate_memory_mb` gives a rough estimate only; measure with
`scripts/benchmark.py` on the real GPU. For the default model it printed ~459 MB (rough, excludes CUDA context and
allocator fragmentation) - this is an estimate, NOT a measurement.
