---
title: Distributed Training
tags:
  - infra
  - training
  - distributed
---

# Distributed Training

Distributed training splits a model or its data across multiple devices to fit larger models or finish faster. The main paradigms are data parallelism, tensor/model parallelism, pipeline parallelism, and sharded optimizers (e.g. ZeRO/FSDP).

## Practical Checks

- Start with data parallelism (DDP); reach for model/pipeline parallel only when memory forces it.
- Verify gradients are synchronized and the effective batch size is what you intend.
- Watch communication overhead — interconnect bandwidth often caps scaling.
- Confirm a single-GPU run matches loss curves before scaling out.
- Checkpoint regularly so a failed node does not lose a long run.

## Related

- [[infra/gpu|GPU]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/index|Infra]]
