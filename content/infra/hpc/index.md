---
title: HPC
tags:
  - infra
  - hpc
---

# HPC

HPC note는 shared compute에서 반복되는 scheduling, resource request, Slurm, distributed training, job array, checkpointing, recovery workflow를 public-safe하게 정리합니다.

이 노트들은 generic하게 유지해야 합니다. Private cluster name, hostname, account name, queue name, SSH detail, internal path, unpublished run result는 공개하지 않습니다.

## Core Workflow

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/distributed-training|Distributed training on HPC]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/hpc/job-array|Job array]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/preemption-resume|Preemption and resume]]

## Job Lifecycle

HPC note는 job이 어디에서 멈췄는지 먼저 분류합니다.

| Stage | Main Question |
| --- | --- |
| Submit | script, account, partition, dependency, array shape가 유효한가? |
| Pending | resource request가 cluster state와 맞는가? |
| Running | CPU/GPU/memory/IO가 의도한 대로 쓰이는가? |
| Checkpoint | walltime, preemption, crash 후 resume 가능한가? |
| Finish | output, logs, artifacts, exit code가 reconciliation 되었는가? |

## Related Systems

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/hardware/index|Hardware]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/gpu/index|GPU]]
- [[infra/gpu/index#memory|GPU memory]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/reproducibility/run-record|Reproducible run record]]

## Checks

- Is the workload CPU-bound, GPU-bound, memory-bound, IO-bound, or scheduler-bound?
- Is the resource request measured from a smoke run?
- Does distributed training need one node, several nodes, or only larger single-node batches?
- Can failed or preempted work resume without corrupting outputs?
- Is every submitted job reconciled before relaunching or reporting completion?
- Are public notes stripped of private infrastructure details?

## Related

- [[infra/index|Infra]]
- [[concepts/systems/index|AI systems]]
- [[projects/hpc-research-workflows|HPC research workflows]]
