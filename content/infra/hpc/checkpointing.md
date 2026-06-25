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
- During reconciliation, treat a checkpoint as resumable only after compatibility checks pass.

## Completion Markers

For long jobs, a checkpoint is not the same as a completed output. Use an explicit marker or manifest:

$$
\text{complete}
\ne
\text{latest checkpoint exists}
$$

The manifest should identify the final step, expected shard count, config hash, and artifact type without exposing private paths.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[infra/distributed-training|Distributed training]]
- [[infra/index|Infra]]
