---
title: Projects
tags:
  - projects
---

# Projects

Project page는 목표, interface, 설계 결정, 검증 방법, 공개 artifact를 설명합니다. 이 영역은 아이디어 자체보다 실제로 만들거나 운영하거나 공개 산출물로 남기는 것에 둡니다.

## 역할

이 영역은 만들고 있거나, 운영 중이거나, 반복 개선 중인 artifact와 workflow를 둡니다. Project page는 문제를 artifact, 설계 결정, 검증 방법, 현재 상태, 다음 공개 개선점으로 연결해야 합니다.

재사용 가능한 정의만 있으면 [[concepts/index|Concepts]]에 둡니다. 특정 논문 하나가 중심이면 [[papers/index|Papers]]에 둡니다. project artifact 없이 운영 경험만 정리한 글이면 [[infra/index|Infra]] 또는 [[logs/index|Logs]]에 둡니다. 질문, 가설, 연구 방향이 중심이면 먼저 [[research/index|Research]]에 둡니다.

## Research와의 차이

| 페이지 중심 | 둘 곳 |
| --- | --- |
| public question, hypothesis, research idea | [Research](/research) |
| implemented pipeline, tool, artifact, workflow | [Projects](/projects) |
| paper-specific claim | [Papers](/papers) |
| reusable concept or formula | [Concepts](/concepts) |
| operational lesson without an artifact | [Infra](/infra) or [Logs](/logs) |

## 묶음

| 묶음 | 용도 |
| --- | --- |
| Research systems | pipelines, experiment workflows, reproducible artifacts |
| Structure-based modeling | docking, scoring, pose quality, screening workflows |
| Protein modeling | representation, structure, sequence-structure tools |
| HPC / server utilities | Slurm, GPU, storage, environment workflow tools |
| Knowledge-base tooling | blog/wiki structure, paper curation, note promotion |
| Agent workflows | coding, paper briefs, verification, orchestration |

## Project Notes

| 프로젝트 | 초점 |
| --- | --- |
| [LLM Wiki blog](/projects/llm-wiki-blog) | public knowledge-base structure |
| [Paper brief agent pipeline](/projects/paper-brief-agent-pipeline) | paper discovery and curation workflow |
| [HPC research workflows](/projects/hpc-research-workflows) | reproducible research engineering on shared compute |

## 공개 상태

| Status | 의미 |
| --- | --- |
| `active` | 실제로 운영하거나 계속 개선 중인 public-safe artifact가 있음 |
| `draft` | artifact boundary는 있지만 검증이나 공개 범위가 더 필요함 |
| `idea` | 아직 project page가 아니라 후보 질문 또는 방향 |
| `archived` | 더 이상 진행하지 않지만 reference로 보관 |

## 형식

| 필요 | 형식 |
| --- | --- |
| Project state | [Project lifecycle](/projects/project-lifecycle), [Project note format](/projects/project-note-format) |
| Milestone or release | [Project milestone format](/projects/project-milestone-format), [Project artifact release](/projects/project-artifact-release) |
| Model or interface | [Model card](/concepts/systems/model-card), [Inference contract](/concepts/systems/inference-contract) |
| Research decision | [Decision record](/concepts/research-methodology/decision-record), [Experiment ledger](/concepts/research-methodology/experiment-ledger) |
| Paper reproduction | [Implementation readiness](/papers/reproducibility/implementation-readiness), [Reproduction result](/papers/reproducibility/reproduction-result) |

## Project Seeds

아래 항목은 아직 project note가 아닙니다. 구현물, workflow, public artifact boundary가 생기면 Projects로 승격하고, 질문만 남아 있으면 [[research/index|Research]]에 둡니다.

| Seed | 방향 |
| --- | --- |
| Structure-based screening pipeline | connect SBDD workflow, scoring, and evaluation |
| Protein-modeling experiment template | standardize representation, split, metric, and run records |
| Slurm/GPU monitoring utility | public runbook and dashboard design notes |
| Agent workflow notes | research engineering and verification workflows |

## Related

| 영역 | 링크 |
| --- | --- |
| Research | [Research](/research), [Research methodology](/concepts/research-methodology) |
| Systems | [Run artifact](/concepts/systems/run-artifact), [Slurm](/infra/hpc/slurm) |
| Papers | [Papers](/papers) |
| Logs and inbox | [Public logs](/logs), [Public log format](/logs/public-log-format), [Publishing gate](/inbox/publishing-gate) |
| Agents | [Agents](/agents) |
