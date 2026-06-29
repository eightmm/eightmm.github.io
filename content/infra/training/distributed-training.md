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

Keep the infra version focused on scheduler, GPU, network, checkpoint, and failure-recovery behavior. Optimizer, loss, objective, and model-parallel design belong in AI or systems notes.
