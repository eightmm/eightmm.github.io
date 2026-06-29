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

The tradeoff should be judged by the actual constraint:

$$
\text{objective}
=
\min
\{\text{peak memory},\ \text{wall time},\ \text{cost},\ \text{latency},\ \text{failure risk}\}
$$

Optimizing one term can make another worse.

## Examples

- Activation checkpointing saves memory by recomputing activations.
- Gradient accumulation increases effective batch size without fitting the full batch at once.
- Quantization reduces parameter and cache memory.
- Caching avoids recomputation but consumes memory.
- Sharding distributes parameters, optimizer state, or activations across devices.

## Technique Map

| Technique | Saves | Pays With | Watch |
| --- | --- | --- | --- |
| Activation checkpointing | activation memory | recomputation | longer step time |
| Gradient accumulation | batch memory | more micro-steps | optimizer-step accounting |
| Quantization | parameter/cache memory | accuracy or kernel constraints | calibration and hardware support |
| KV cache | decode compute | memory capacity | context length and concurrency |
| Sharding | per-device memory | communication | bandwidth and failure complexity |
| Caching embeddings/features | repeated compute | storage and stale artifacts | data-version alignment |

## Checks

- Is the bottleneck capacity, bandwidth, compute, or communication?
- Are parameters, activations, optimizer state, gradients, or KV cache dominant?
- Does the technique change numerical behavior or only resource use?
- Is wall-clock time, cost, or maximum problem size the real constraint?
- Does a smaller model or shorter context solve the problem more simply?
- Are comparisons made at fixed quality, fixed latency, fixed cost, or fixed memory?
- Does the run record state which resource tradeoff was intentionally chosen?

## Related

- [[infra/gpu/index#memory|GPU memory]]
- [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[concepts/systems/inference-capacity-planning|Inference capacity planning]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
