---
title: Memory-Compute Tradeoff
tags:
  - systems
  - gpu
  - performance
---

# Memory-Compute Tradeoff

Memory-compute tradeoff describes techniques that reduce memory use by spending more computation, or reduce computation by spending more memory. It appears in training, inference, retrieval, and data loading.

A rough training memory budget is:

$$
M_{\mathrm{train}}
\approx M_{\mathrm{params}} + M_{\mathrm{grads}} + M_{\mathrm{opt}} + M_{\mathrm{activations}}
$$

For autoregressive inference, KV cache can dominate:

$$
M_{\mathrm{KV}} \propto L \cdot H \cdot d_{\mathrm{head}} \cdot n_{\mathrm{layers}}
$$

where $L$ is sequence length, $H$ is the number of attention heads, and $d_{\mathrm{head}}$ is head dimension.

## Examples

- Activation checkpointing saves memory by recomputing activations.
- Gradient accumulation increases effective batch size without fitting the full batch at once.
- Quantization reduces parameter and cache memory.
- Caching avoids recomputation but consumes memory.
- Sharding distributes parameters, optimizer state, or activations across devices.

## Checks

- Is the bottleneck capacity, bandwidth, compute, or communication?
- Are parameters, activations, optimizer state, gradients, or KV cache dominant?
- Does the technique change numerical behavior or only resource use?
- Is wall-clock time, cost, or maximum problem size the real constraint?
- Does a smaller model or shorter context solve the problem more simply?

## Related

- [[infra/gpu/memory|GPU memory]]
- [[infra/gpu/bottleneck-taxonomy|GPU bottleneck taxonomy]]
- [[infra/training/distributed-training|Distributed training]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
