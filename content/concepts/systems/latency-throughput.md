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

## Queueing Boundary

Observed latency often includes queueing:

$$
t_{\mathrm{latency}}
=
t_{\mathrm{queue}}
+
t_{\mathrm{preprocess}}
+
t_{\mathrm{model}}
+
t_{\mathrm{postprocess}}
+
t_{\mathrm{transfer}}
$$

Optimizing model compute alone may not improve user-visible latency if queueing, preprocessing, or transfer dominates.

## Autoregressive Models

For decoder-only generation, separate prefill and decode:

$$
t_{\mathrm{total}}
\approx
t_{\mathrm{prefill}}(L_{\mathrm{in}})
+
T_{\mathrm{out}}
\cdot
t_{\mathrm{decode}}
$$

where $L_{\mathrm{in}}$ is input length and $T_{\mathrm{out}}$ is generated output length. Tokens/sec should state whether it measures prefill, decode, or end-to-end throughput.

## Benchmark Design

A performance benchmark should define:

$$
(\text{request distribution},
\text{batching policy},
\text{concurrency},
\text{input size},
\text{output size},
\text{hardware class},
\text{quality constraints})
$$

Otherwise latency and throughput numbers are not comparable.

## Checks

- Is the target p50, p95, p99, average latency, tokens/sec, samples/sec, or jobs/day?
- Are prefill and decode measured separately for autoregressive models?
- Is the benchmark using realistic request sizes?
- Are cold-start, queue time, preprocessing, and postprocessing included?
- Does the metric match the user-facing workflow?
- Is quality held constant when comparing speed optimizations?
- Are failed, timed-out, or truncated requests included in the denominator?
- Is throughput limited by compute, memory bandwidth, storage IO, network, or scheduler queueing?

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/inference-serving|Inference serving]]
- [[concepts/systems/inference-capacity-planning|Inference capacity planning]]
- [[infra/gpu/index#memory|GPU memory]]
