---
title: Distributed Training
aliases:
  - infra/distributed-training
tags:
  - infra
  - training
  - distributed
---

# Distributed Training

Distributed training splits a model or its data across multiple devices to fit larger models or finish faster. The conceptual layer is [[concepts/systems/distributed-training|Distributed training]]; this note focuses on public operational checks.

## Practical Checks

- Start with data parallelism (DDP); reach for model/pipeline parallel only when memory forces it.
- Verify gradients are synchronized and the effective batch size is what you intend.
- Watch communication overhead — interconnect bandwidth often caps scaling.
- Confirm a single-GPU run matches loss curves before scaling out.
- Checkpoint regularly so a failed node does not lose a long run.
- Use [[infra/gpu/bottleneck-taxonomy|GPU bottleneck taxonomy]] to separate single-device bottlenecks from communication bottlenecks.

## Related

- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[infra/gpu/index|GPU]]
- [[infra/gpu/bottleneck-taxonomy|GPU bottleneck taxonomy]]
- [[infra/gpu/memory|GPU memory]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/index|Infra]]
