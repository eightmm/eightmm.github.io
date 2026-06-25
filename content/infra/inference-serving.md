---
title: Inference Serving
tags:
  - infra
  - inference
  - serving
---

# Inference Serving

Inference serving turns a trained model into a low-latency, high-throughput service. For LLMs the dominant costs are KV-cache memory and the prefill/decode split; batching and quantization are the main levers.

## Practical Checks

- Distinguish latency-bound (single request) from throughput-bound (batched) goals.
- Use continuous/dynamic batching to keep the GPU busy without hurting tail latency.
- Size the KV cache against context length and concurrency.
- Apply quantization (int8/fp8) only after measuring accuracy impact.
- Measure p50/p95/p99 latency, not just averages.

## Related

- [[infra/gpu|GPU]]
- [[infra/gpu-memory|GPU memory]]
- [[infra/distributed-training|Distributed training]]
- [[concepts/evaluation/calibration|Calibration]]
- [[infra/index|Infra]]
