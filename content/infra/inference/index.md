---
title: Inference
unlisted: true
tags:
  - infra
  - inference
  - serving
---

# Inference Runbooks

Inference runbooks cover serving operations under memory, latency, throughput, and reliability constraints. Start with [[concepts/systems/inference|Inference]], [[concepts/systems/model-serving|Model serving]], and [[concepts/systems/latency-throughput|Latency and throughput]] for the reusable systems concepts.

Serving은 interface와 capacity contract에서 시작합니다.

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

- [[concepts/systems/inference-serving|Inference serving]]
- [[concepts/systems/inference-capacity-planning|Inference capacity planning]]

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
