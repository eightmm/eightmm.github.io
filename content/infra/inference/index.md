---
title: Inference
tags:
  - infra
  - inference
  - serving
---

# Inference

Inference infra notes cover how models are served under memory, latency, throughput, and reliability constraints.

Serving starts from an interface and capacity contract:

$$
(\text{input schema}, \text{output schema}, \text{latency target}, \text{throughput target}, \text{memory budget})
\rightarrow
\text{serving policy}
$$

The serving policy includes batching, cache management, quantization, fallback behavior, and monitoring boundaries.

## Scope

- Online and batch serving of models.
- Capacity planning for request shape, concurrency, context length, and memory.
- Latency/throughput tradeoffs and tail-latency measurement.
- Public-safe inference contracts that do not expose private prompts, datasets, or endpoints.

## Notes

- [[infra/inference/serving|Inference serving]]
- [[infra/inference/capacity-planning|Inference capacity planning]]

## Checks

- Is the task latency-bound, throughput-bound, memory-bound, or quality-bound?
- Are prefill and decode separated when autoregressive models are involved?
- Is KV-cache memory included in the capacity plan?
- Are p50, p95, and p99 latency treated differently from average latency?
- Are request logs and examples sanitized before becoming public notes?

## Where New Notes Go

- Serving architecture and batching go here.
- Model-side output contracts go under [[concepts/systems/inference-contract|Inference contract]].
- GPU capacity and memory diagnosis go under [[infra/gpu/index|GPU]].
- General deployment patterns go under [[concepts/systems/deployment-strategy|Deployment strategy]].

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[infra/gpu/index|GPU]]
