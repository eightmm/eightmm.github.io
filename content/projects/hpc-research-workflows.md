---
title: HPC Research Workflows
tags:
  - projects
  - infra
  - hpc
---

# HPC Research Workflows

이 project는 shared GPU/HPC system에서 research workload를 실행할 때 필요한 공개 가능한 일반 패턴을 모읍니다. Private cluster detail이 아니라 reproducibility, debugging, safe operation에 초점을 둡니다.

## Artifact

Artifact는 research compute run을 설계, 제출, 모니터링, 종료하는 재사용 가능한 workflow입니다.

## Status

Lifecycle status: `draft`.

This is a public workflow note, not a released software package. The artifact is the reusable checklist and note structure for safe research compute work.

## Problem

Research run은 model 자체보다 주변 workflow가 약해서 실패하는 경우가 많습니다. 예를 들면 resource request가 불명확하거나, checkpoint가 없거나, environment가 추적되지 않거나, failure log가 부족한 경우입니다.

## Public Boundary

Note는 일반화된 형태로 유지합니다. 실제 server name, IP, account name, SSH port, private mount path, infrastructure를 드러내는 queue name, user list, private job ID는 넣지 않습니다.

## Workflow

1. Local 또는 small-batch smoke test를 먼저 실행합니다.
2. CPU, GPU, memory, time assumption을 명시한 constrained job을 제출합니다.
3. Checkpoint와 log를 재현 가능한 layout으로 저장합니다.
4. Code commit, environment, seed, dataset version을 기록합니다.
5. Scheduler state, application log, hardware symptom을 기준으로 failure를 디버깅합니다.

## Artifact Release

- Run record schema: public.
- Generic Slurm/HPC pattern: public.
- Private cluster topology, account name, job ID, hostname, path, live metric: not released.
- Unpublished experiment result: public-safe method note로 변환하기 전에는 not released.

| Artifact | Release status |
| --- | --- |
| Generic workflow checklist | released |
| Example run record fields | released |
| Cluster-specific scripts, hostnames, queues, paths | not released |
| Private experiment logs or metrics | not released |

## Checks

- private infrastructure detail 없이 public information만으로 run을 재현할 수 있는가?
- resource assumption이 failure를 설명할 만큼 명시되어 있는가?
- checkpoint 주기가 expected wall time에 비해 충분한가?
- 결과가 public method note, private experiment, post candidate 중 무엇인가?
- 공개하는 artifact가 [[projects/project-artifact-release|Project artifact release]] 기준으로 sanitized되었는가?

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-artifact-release|Project artifact release]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/project-note-format|Project note format]]
