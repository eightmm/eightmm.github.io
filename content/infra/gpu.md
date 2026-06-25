---
title: GPU
tags:
  - infra
  - gpu
---

# GPU

GPUs accelerate the dense linear algebra behind training and inference. The recurring constraints are memory capacity, memory bandwidth, and interconnect — not just raw FLOPs.

## Practical Checks

- Track memory headroom: model weights, activations, optimizer state, and KV cache.
- Match precision (bf16/fp16/fp8) to the hardware and numerical tolerance.
- Use TF32/bf16 matmul paths where accuracy allows.
- Profile before optimizing — find whether you are compute-, memory-, or IO-bound.
- Keep public notes generic; do not publish private hosts or allocations.

## Related

- [[infra/gpu-memory|GPU memory]]
- [[infra/distributed-training|Distributed training]]
- [[infra/inference-serving|Inference serving]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/index|Infra]]
