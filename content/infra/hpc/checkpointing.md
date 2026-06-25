---
title: Checkpointing
tags:
  - infra
  - hpc
  - training
---

# Checkpointing

Checkpointing periodically saves training state so a job can resume after preemption, failure, or a wall-clock limit. A checkpoint should restore the run exactly, not just the weights.

## Practical Checks

- Save model weights, optimizer state, scheduler state, step count, and RNG seeds.
- Write atomically (temp file then rename) so a crash never corrupts the latest checkpoint.
- Keep a rolling window plus periodic milestones to bound disk use.
- Test resume on a small run before trusting it on a long one.
- Record code commit, environment, and dataset version alongside the checkpoint.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[infra/distributed-training|Distributed training]]
- [[infra/index|Infra]]
