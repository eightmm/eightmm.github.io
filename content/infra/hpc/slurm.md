---
title: Slurm
tags:
  - infra
  - hpc
  - slurm
---

# Slurm

Slurm is a workload manager used to submit, schedule, monitor, and cancel jobs on shared compute clusters. This page keeps public, non-sensitive workflow notes only.

Slurm note의 핵심은 command 암기가 아니라 job과 scheduler 사이의 contract를 읽는 것입니다.

$$
\text{job}
=
\text{script}
+ \text{resource request}
+ \text{environment}
+ \text{artifacts}
+ \text{exit evidence}
$$

공유 cluster에서는 같은 코드도 resource request, queue state, environment, IO path에 따라 전혀 다른 run이 됩니다. 그래서 Slurm 글은 “어떻게 제출했는가”보다 “무엇을 요청했고, 무엇이 실제로 실행됐고, 무엇으로 완료를 확인했는가”를 남깁니다.

## Public Checklist

- Use generic examples instead of site-specific cluster names.
- Avoid account names, hostnames, SSH connection details, internal paths, and private partitions.
- Record resource assumptions without exposing project-specific allocations.
- Keep unpublished metrics and experiment results out of public notes.

## Generic Commands

명령어는 상태를 직접 증명하지 않습니다. 각 command가 어떤 evidence를 주는지 분리해서 봅니다.

| Command | Use for | Does not prove |
| --- | --- | --- |
| `sinfo` | cluster resource and partition state at query time | your job will start soon |
| `squeue` | pending/running/cancelled job state | output correctness |
| `sbatch job.sbatch` | job submission and job id creation | script will finish successfully |
| `scancel <job-id>` | cancellation request | artifact cleanup or rollback |
| `sacct` | historical job accounting when available | application-level success |
| `tail logs/...` | recent stdout/stderr evidence | complete artifact validity |

```bash
sinfo
squeue
sbatch job.sbatch
scancel <job-id>
```

## Job State Reading

| State question | Start |
| --- | --- |
| What did the script request? | [Resource request](/infra/hpc/resource-request), [Slurm job script](/infra/hpc/slurm-job-script) |
| What limits or accounting policy apply? | [Slurm Accounting and Limits](/infra/hpc/slurm-accounting-limits) |
| Why is the job pending? | [Resource scheduling](/concepts/systems/resource-scheduling) |
| Did the job run with the expected environment? | [Environments](/infra/environments), [Environment modules and containers](/infra/environments/modules-containers) |
| Did the job write complete artifacts? | [Reproducible run record](/infra/reproducibility/run-record), [Job reconciliation](/infra/hpc/job-reconciliation) |
| Can it resume after failure or preemption? | [Checkpointing](/infra/hpc/checkpointing), [Preemption and resume](/infra/hpc/preemption-resume) |

## Common Failure Shapes

| Symptom | First check |
| --- | --- |
| Job stays pending | request size, wall time, dependency, scheduler policy |
| Job starts then fails immediately | shell options, environment activation, missing files, permissions |
| GPU allocated but idle | data loading, device visibility, process launch, CPU bottleneck |
| OOM or killed job | memory request, batch size, checkpoint interval, data shape |
| Output exists but result is suspicious | exit code, manifest, logs, validation, seed/config mismatch |

## Reproducibility Notes

- Capture code commit, environment, seed, and dataset version.
- Prefer small smoke tests before large jobs.
- Keep resource requests explicit and public-safe: CPU count, GPU count, memory class, and wall time can be described generically.
- Use job arrays for independent shards instead of one oversized job when the workload is naturally parallel.
- Design long jobs to checkpoint and resume before wall-time limits or preemption.
- Reconcile terminal job state against logs and artifacts before relaunching or reporting success.
- Link public experiment methodology into [[agents/workflows/llm-wiki|LLM Wiki]] pages when it becomes reusable.

## Related

- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]]
- [[infra/hpc/job-array|Job array]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[projects/index|Project index]]
- [[infra/index|Infra]]
