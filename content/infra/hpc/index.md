---
title: HPC
tags:
  - infra
  - hpc
---

# HPC

HPC notes collect public workflow patterns for shared compute: scheduling, resource requests, job scripts, job arrays, checkpointing, and recovery.

These notes should stay generic. Do not publish private cluster names, hostnames, account names, queue names, SSH details, internal paths, or unpublished run results.

## Core Workflow

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-array|Job array]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/preemption-resume|Preemption and resume]]

## Related Systems

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/gpu|GPU]]
- [[infra/gpu-memory|GPU memory]]
- [[infra/data-loading-io|Data loading and IO]]
- [[infra/reproducible-run-record|Reproducible run record]]

## Checks

- Is the workload CPU-bound, GPU-bound, memory-bound, IO-bound, or scheduler-bound?
- Is the resource request measured from a smoke run?
- Can failed or preempted work resume without corrupting outputs?
- Are public notes stripped of private infrastructure details?

## Related

- [[infra/index|Infra]]
- [[concepts/systems/index|AI systems]]
- [[projects/hpc-research-workflows|HPC research workflows]]
