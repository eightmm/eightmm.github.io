---
title: Preemption and Resume
tags:
  - infra
  - hpc
  - reliability
---

# Preemption and Resume

Preemption is when a running job is stopped before completion because the scheduler needs resources for higher-priority work or because a time/resource limit is reached. Resume is the workflow that continues from a saved state.

For a long job:

$$
\text{progress}_{t+1}
= \operatorname{resume}(\text{checkpoint}_t, \text{inputs}, \text{config})
$$

The checkpoint must contain enough state to avoid silently changing the run after restart.

## What To Save

- Model weights and optimizer state for training.
- Step, epoch, random seeds, and scheduler state.
- Data shard or task index.
- Config, code commit, and environment summary.
- Partial outputs with completion markers.

## Resume Pattern

1. Start from a clean run record.
2. Write checkpoints atomically or with temporary files plus rename.
3. Validate checkpoint compatibility before loading.
4. Resume from the latest complete checkpoint.
5. Record whether the final output came from a resumed run.
6. Reconcile the resumed job before marking the run complete.

## Checks

- Can the job restart after losing the process?
- Does the checkpoint include optimizer and scheduler state when training?
- Are partial outputs distinguishable from complete outputs?
- Is the resume path tested with a short smoke run?
- Does public documentation avoid private job IDs, paths, queues, and result details?
- Does the run record distinguish original launch, interruption, resume, and final closeout?

## Related

- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/reproducibility/run-record|Reproducible run record]]
