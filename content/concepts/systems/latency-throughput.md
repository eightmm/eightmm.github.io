---
title: Latency and Throughput
tags:
  - systems
  - inference
  - performance
---

# Latency and Throughput

Latency measures how long one request takes. Throughput measures how much work a system completes per unit time. Optimizing one can hurt the other.

For requests completed in a time window $\Delta t$:

$$
\operatorname{throughput}=\frac{N_{\mathrm{completed}}}{\Delta t}
$$

Latency is a distribution, not one number. Tail latency is often more important than the mean:

$$
p95 = \operatorname{Quantile}_{0.95}(\{t_i\}_{i=1}^{N})
$$

## Common Tradeoffs

- Larger batches improve throughput but can increase latency.
- Caching reduces repeated work but uses memory.
- Quantization can reduce latency and memory, but may affect quality.
- More workers can improve concurrency until memory, I/O, or scheduling becomes the bottleneck.
- Streaming can reduce perceived latency even if total generation time is unchanged.

## Checks

- Is the target p50, p95, p99, average latency, tokens/sec, samples/sec, or jobs/day?
- Are prefill and decode measured separately for autoregressive models?
- Is the benchmark using realistic request sizes?
- Are cold-start, queue time, preprocessing, and postprocessing included?
- Does the metric match the user-facing workflow?

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[infra/inference-serving|Inference serving]]
- [[infra/inference-capacity-planning|Inference capacity planning]]
- [[infra/gpu-memory|GPU memory]]
