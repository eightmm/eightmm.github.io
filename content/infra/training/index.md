---
title: Training Infra
unlisted: true
tags:
  - infra
  - training
---

# Training Runbooks

Training runbooks connect training process concepts to resource and operation constraints. Start with [[concepts/systems/training-run|Training run]] for reusable systems concepts. Cluster-level launch, scheduler allocation, and multi-node execution belong under [[infra/hpc/distributed-training|Distributed training on HPC]].

운영 관점의 training loop는 model update보다 넓습니다.

$$
\text{train step}
\rightarrow
\text{checkpoint}
\rightarrow
\text{validation}
\rightarrow
\text{resume or select}
$$

여기 있는 infra note는 device topology, process launch, synchronization, checkpoint survival, stability evidence처럼 이 loop 주변의 machinery에 집중합니다.

## Scope

- Single-node training runbooks and scaling checks.
- Effective batch size, gradient synchronization, and checkpoint state from an operational perspective.
- Failure recovery for long jobs when the failure pattern is generic and public-safe.
- Links between training systems and [[infra/hpc/index|HPC]] scheduling.

## Notes

- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[infra/hpc/distributed-training|Distributed training on HPC]]

## Checks

- Does a single-device run match the expected loss behavior before scaling out?
- Is the effective batch size defined after accumulation and data parallelism?
- Are checkpoint contents sufficient to resume optimizer, scheduler, scaler, and RNG state?
- Is the bottleneck compute, memory, IO, synchronization, communication, or scheduler wait?
- Are logs public-safe and free of hostnames, usernames, queue names, and private paths?

## Routing

| Question | Go To |
| --- | --- |
| What is the reusable training process? | [Training run](/concepts/systems/training-run) |
| What must be saved to resume? | [Checkpoint state](/concepts/systems/checkpoint-state), [Checkpointing](/infra/hpc/checkpointing) |
| Is this scheduler-managed multi-node training? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| Is this GPU memory or utilization diagnosis? | [GPU](/infra/gpu) |
| Is this optimizer or objective math? | [Optimizer](/concepts/machine-learning/optimizer), [Machine Learning](/ai/machine-learning) |

## Related

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[infra/gpu/index|GPU]]
- [[infra/hpc/index|HPC]]
