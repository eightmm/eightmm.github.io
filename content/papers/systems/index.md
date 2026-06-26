---
title: Systems Papers
tags:
  - papers
  - systems
  - infra
---

# Systems Papers

Systems paper notes cover infrastructure, training efficiency, inference serving, distributed execution, agent tooling, and reproducibility.

## Reading Axes

- What bottleneck is addressed: memory, compute, communication, I/O, latency, or reliability?
- What is the system boundary: kernel, runtime, framework, scheduler, serving stack, or workflow?
- What workload is evaluated?
- Are baselines and hardware assumptions stated clearly?
- Can the result be generalized beyond the reported setup?
- Is the claim about latency, throughput, memory, cost, reliability, or developer workflow?

## Concepts

- [[concepts/systems/index|AI systems]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[infra/gpu/index#memory|GPU memory]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/training/distributed-training|Distributed training]]
- [[infra/inference/serving|Inference serving]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/verification/agent-evaluation|Agent evaluation]]

## Related

- [[infra/index|Infra]]
- [[agents/index|Agents]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
