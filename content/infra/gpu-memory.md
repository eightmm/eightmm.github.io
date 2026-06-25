---
title: GPU Memory
tags:
  - infra
  - gpu
  - training
---

# GPU Memory

GPU memory is usually the first hard limit in training and inference. A run can be compute-light but still fail if weights, activations, optimizer state, gradients, or KV cache exceed device memory.

For training, a rough memory decomposition is:

$$
M_{\mathrm{total}}
\approx
M_{\mathrm{weights}}
+ M_{\mathrm{gradients}}
+ M_{\mathrm{optimizer}}
+ M_{\mathrm{activations}}
+ M_{\mathrm{buffers}}
$$

For autoregressive inference, KV cache often dominates:

$$
M_{\mathrm{KV}}
\approx
2 \cdot L \cdot H \cdot T \cdot d_{\mathrm{head}} \cdot b
$$

where $L$ is number of layers, $H$ is number of attention heads, $T$ is context length, and $b$ is bytes per value.

## Checks

- Is memory used by parameters, activations, optimizer state, batch size, or KV cache?
- Does memory grow every step, suggesting a graph retention or logging leak?
- Is mixed precision enabled and numerically appropriate?
- Would gradient checkpointing, smaller batch size, sharded optimizer, or shorter context solve the bottleneck?
- Are measurements taken from the same process and device that owns the allocation?

## Related

- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[infra/gpu|GPU]]
- [[infra/distributed-training|Distributed training]]
- [[infra/inference-serving|Inference serving]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
