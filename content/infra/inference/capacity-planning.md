---
title: Inference Capacity Planning
aliases:
  - infra/inference-capacity-planning
tags:
  - infra
  - inference
  - serving
  - gpu
---

# Inference Capacity Planning

Inference capacity planning estimates how many requests a model can serve under a latency, throughput, memory, and quality target. It connects [[infra/inference/serving|Inference serving]], [[infra/gpu/memory|GPU memory]], and [[concepts/systems/latency-throughput|Latency and throughput]].

A simple capacity constraint is:

$$
N_{\mathrm{concurrent}}
\le
\left\lfloor
\frac{M_{\mathrm{available}} - M_{\mathrm{weights}} - M_{\mathrm{runtime}}}
{M_{\mathrm{request}}}
\right\rfloor
$$

where $M_{\mathrm{request}}$ includes activation, KV cache, buffers, and request-specific state.

For autoregressive models:

$$
t_{\mathrm{request}}
\approx
t_{\mathrm{prefill}}(T_{\mathrm{in}})
+ T_{\mathrm{out}} \cdot t_{\mathrm{decode}}
$$

where $T_{\mathrm{in}}$ is input context length and $T_{\mathrm{out}}$ is generated output length.

## Planning Fields

- Workload: request rate, input length, output length, batch shape, and concurrency.
- Resource envelope: GPU memory, CPU preprocessing, storage, network, and runtime overhead.
- Serving objective: p50, p95, p99 latency, tokens/sec, samples/sec, or jobs/day.
- Quality boundary: quantization, truncation, batching, caching, and fallback impact.
- Failure policy: timeout, queue overflow, unsupported input, low confidence, or retry.
- Logging boundary: public-safe metrics only, no private prompts, paths, credentials, or unpublished data.

## Checks

- Are prefill and decode measured separately when relevant?
- Is the batch policy static, dynamic, continuous, or offline batch?
- Is capacity estimated from realistic context lengths and output lengths?
- Does quantization preserve the evaluation boundary and intended use?
- Are tail latency, error rate, and queue time measured together?
- Is the serving contract documented with [[concepts/systems/inference-contract|Inference contract]]?

## Related

- [[infra/inference/serving|Inference serving]]
- [[infra/gpu/index|GPU]]
- [[infra/gpu/memory|GPU memory]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/inference-contract|Inference contract]]
