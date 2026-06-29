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

| Stage | Main question |
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

- workload가 CPU-bound, GPU-bound, memory-bound, IO-bound, scheduler-bound 중 무엇인가?
- resource request가 smoke run에서 측정된 값인가?
- distributed training에 one node, several nodes, larger single-node batch 중 무엇이 필요한가?
- failed 또는 preempted work가 output을 corrupt하지 않고 resume될 수 있는가?
- relaunch 또는 completion report 전에 submitted job이 모두 reconciled되었는가?
- public note에서 private infrastructure detail을 제거했는가?

## Related

- [[infra/index|Infra]]
- [[concepts/systems/index|AI systems]]
- [[projects/hpc-research-workflows|HPC research workflows]]
