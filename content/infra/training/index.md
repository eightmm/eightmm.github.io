---
title: Training Infra
unlisted: true
tags:
  - infra
  - training
---

# Training Runbooks

Training runbooks cover operational choices around training loop stability, checkpoints, single-node scaling checks, and run behavior. The canonical concept entry is [[concepts/systems/training-run|Training run]]. Cluster-level launch, scheduler allocation, and multi-node execution belong under [[infra/hpc/distributed-training|Distributed training on HPC]].

The operational training loop is broader than the model update:

$$
\text{train step}
\rightarrow
\text{checkpoint}
\rightarrow
\text{validation}
\rightarrow
\text{resume or select}
$$

Infra notes here focus on the machinery around that loop: device topology, process launch, synchronization, checkpoint survival, and stability evidence.

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

## Where New Notes Go

- Single-node multi-GPU checks and training loop behavior go here.
- Slurm launch, node layout, scheduler allocation, and multi-node training go under [[infra/hpc/distributed-training|Distributed training on HPC]].
- Slurm job script shape goes under [[infra/hpc/index|HPC]].
- GPU memory and utilization diagnosis goes under [[infra/gpu/index|GPU]].
- Optimizer math stays under [[concepts/machine-learning/optimizer|Optimizer]].

## Related

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[infra/gpu/index|GPU]]
- [[infra/hpc/index|HPC]]
