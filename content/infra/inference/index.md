---
title: Inference
unlisted: true
tags:
  - infra
  - inference
  - serving
---

# Inference Runbooks

Inference runbooks connect serving concepts to resource and operation constraints. Start with [[concepts/systems/inference|Inference]], [[concepts/systems/model-serving|Model serving]], and [[concepts/systems/latency-throughput|Latency and throughput]] for reusable systems concepts; use infra pages when hardware, GPU memory, IO, or operations are the bottleneck.

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

## Routing

| Question | Go To |
| --- | --- |
| What is inference or serving? | [Inference](/concepts/systems/inference), [Model serving](/concepts/systems/model-serving) |
| How should input/output and errors be exposed? | [Inference contract](/concepts/systems/inference-contract) |
| Is the issue latency, throughput, batching, or capacity? | [Inference serving](/concepts/systems/inference-serving), [Inference capacity planning](/concepts/systems/inference-capacity-planning) |
| Is the issue GPU memory or utilization? | [GPU](/infra/gpu) |
| Is the issue deployment policy? | [Deployment strategy](/concepts/systems/deployment-strategy) |

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[infra/gpu/index|GPU]]
