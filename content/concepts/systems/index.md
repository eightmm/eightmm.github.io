---
title: AI Systems
tags:
  - systems
  - infra
  - machine-learning
---

# AI Systems

AI systems notes describe how models are trained, served, measured, and reproduced. They sit between model concepts and infrastructure operations.

The core systems question is:

$$
\text{quality} = f(\text{data}, \text{model}, \text{compute}, \text{workflow}, \text{evaluation})
$$

A model is not only a function $f_\theta$; it is also a training process, an inference process, and an operational artifact.

## Topics

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]

## Checks

- Is the bottleneck data loading, compute, memory, communication, or scheduling?
- Is the workload shaped for the scheduler: resource request, job size, queue time, and preemption risk?
- Is the environment and storage path part of the run record?
- Is the goal model quality, time-to-train, cost, latency, throughput, or reliability?
- Can the run be reproduced from a commit, config, seed, dataset version, and environment?
- Can the workflow recover from preemption, partial output, or service failure?
- Can terminal runs be reconciled from logs, scheduler state, and artifacts?
- Are training metrics, validation metrics, and serving metrics separated?
- Does the deployment path preserve the same preprocessing and constraints as evaluation?

## Related

- [[ai/index|AI]]
- [[infra/index|Infra]]
- [[papers/systems/index|Systems papers]]
- [[concepts/evaluation/index|Evaluation]]
- [[agents/verification/verification-loop|Verification loop]]
