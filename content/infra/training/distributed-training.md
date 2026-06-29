---
title: Distributed Training
unlisted: true
tags:
  - infra
  - training
---

# Distributed Training

Distributed training has two different homes in this wiki:

| Question | Go To |
| --- | --- |
| What is the training-side checklist? | [Distributed training runbook](/concepts/systems/distributed-training-runbook) |
| What does distributed training mean conceptually? | [Distributed training](/concepts/systems/distributed-training) |
| How does a scheduler-managed multi-node job run? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| Which resource request should be checked? | [Resource request](/infra/hpc/resource-request) |

At the infra layer, the important equation is not the loss; it is the scaling budget:

$$
T_{\mathrm{step}}
\approx
T_{\mathrm{compute}}
+ T_{\mathrm{communication}}
+ T_{\mathrm{input}}
+ T_{\mathrm{sync}}
$$

If adding GPUs does not reduce $T_{\mathrm{step}}$, the limiting term is probably communication, data loading, synchronization, checkpointing, or scheduler placement rather than model math.

## Infra Checklist

| Check | Why |
| --- | --- |
| Rank layout | wrong node/GPU mapping creates idle devices or slow communication |
| Network path | all-reduce and parameter sync depend on interconnect topology |
| Data path | shared storage and dataloader workers can dominate step time |
| Checkpoint policy | restart cost and preemption behavior define operational reliability |
| Logs and artifacts | run records need rank, world size, commit, config, and environment |

## Boundary

| Belongs here | Belongs elsewhere |
| --- | --- |
| scheduler request, rank placement, GPU visibility, network, checkpoint restart | loss, optimizer, architecture, objective |
| failed multi-node launch, NCCL/runtime environment, data path saturation | paper claim, metric, dataset semantics |

Keep the infra version focused on scheduler, GPU, network, checkpoint, and failure-recovery behavior. Optimizer, loss, objective, and model-parallel design belong in AI or systems notes.
