---
title: HPC Job Lifecycle
tags:
  - infra
  - hpc
  - workflows
---

# HPC Job Lifecycle

HPC job lifecycle은 workload가 local command에서 scheduled job, running process, checkpointed artifact, final result로 이동하는 과정을 설명합니다.

## 단계

1. code, environment, input data, output layout을 준비합니다.
2. 작은 workload로 smoke test를 실행합니다.
3. explicit resource assumption과 함께 job을 submit합니다.
4. scheduler state, log, GPU use, storage growth를 monitor합니다.
5. checkpoint와 resumable output을 저장합니다.
6. completion, failure, cancellation을 public 또는 private run record로 reconcile합니다.

## State Model

Job state는 scheduler 상태와 artifact 상태를 함께 봐야 합니다.

| Scheduler state | Artifact question | Next action |
| --- | --- | --- |
| pending | resource request가 합리적인가? | request, partition, dependency 확인 |
| running | expected throughput과 storage growth가 보이는가? | monitor and sample logs |
| completed | output과 marker가 완전한가? | reconcile and record |
| failed | failure class가 분류됐는가? | diagnose before relaunch |
| cancelled/preempted | resume 가능한 checkpoint가 있는가? | resume or mark superseded |

Scheduler가 `COMPLETED`를 보여도 output이 complete라는 뜻은 아닙니다. Artifact closeout이 별도로 필요합니다.

## 최소 public record

- workload의 purpose.
- high-level software stack.
- generic term으로 표현한 resource class.
- 공개 가능한 경우 failure mode와 fix.
- private run directory가 아니라 reusable note link.

## Failure class

- Scheduler issue: pending, preempted, time limit, resource mismatch.
- Runtime issue: dependency, CUDA, memory, data loading, shape error.
- Data issue: missing file, corrupted input, split leakage, inconsistent label.
- Storage issue: quota, slow I/O, checkpoint write failure.

## Launch Contract

Launch 전에 아래를 고정합니다.

| Field | Public-safe wording |
| --- | --- |
| workload | training, inference, docking batch, evaluation, preprocessing |
| resource class | CPU, single GPU, multi GPU, memory-heavy, IO-heavy |
| expected artifact | checkpoint, metrics table, generated candidates, logs |
| resume rule | fresh, resume from checkpoint, rerun failed shards |
| stop rule | time limit, convergence, all shards complete, smoke test only |

Private queue names, hostnames, user names, paths, and project identifiers are not part of the public record.

## Closeout

Job은 assumption이 아니라 evidence로 close해야 합니다.

$$
\operatorname{closed}(j)
=
\operatorname{terminal}(j)
\land
\operatorname{artifact\_checked}(j)
\land
\operatorname{recorded}(j)
$$

어떤 항이라도 false라면 next action은 fresh launch가 아니라 reconciliation입니다.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/gpu/index|GPU]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[projects/hpc-research-workflows|HPC research workflows]]
