---
title: HPC Job Lifecycle
tags:
  - infra
  - hpc
  - workflows
---

# HPC Job Lifecycle

An HPC job lifecycle describes how a workload moves from a local command to a scheduled job, running process, checkpointed artifact, and final result.

## Stages

1. Prepare code, environment, input data, and output layout.
2. Run a smoke test with a small workload.
3. Submit the job with explicit resource assumptions.
4. Monitor scheduler state, logs, GPU use, and storage growth.
5. Save checkpoints and resumable outputs.
6. Reconcile completion, failure, or cancellation into a public or private run record.

## Minimal Public Record

- Purpose of the workload.
- Software stack at a high level.
- Resource class in generic terms.
- Failure mode and fix, if publishable.
- Links to reusable notes, not private run directories.

## Failure Classes

- Scheduler issue: pending, preempted, time limit, resource mismatch.
- Runtime issue: dependency, CUDA, memory, data loading, shape error.
- Data issue: missing files, corrupted input, split leakage, inconsistent labels.
- Storage issue: quota, slow I/O, checkpoint write failure.

## Closeout

A job should be closed by evidence, not by assumption:

$$
\operatorname{closed}(j)
=
\operatorname{terminal}(j)
\land
\operatorname{artifact\_checked}(j)
\land
\operatorname{recorded}(j)
$$

If any term is false, the next action is reconciliation, not a fresh launch.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/gpu/index|GPU]]
- [[infra/training/distributed-training|Distributed training]]
- [[projects/hpc-research-workflows|HPC research workflows]]
