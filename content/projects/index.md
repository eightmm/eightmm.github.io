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

## Project Boundary

Project로 승격하려면 반복해서 실행하거나 공개 artifact로 설명할 수 있는 단위가 있어야 합니다. 단순 아이디어, 읽은 논문, 개념 설명만으로는 project가 아닙니다.

$$
\text{project}
=
\text{goal}
+ \text{artifact}
+ \text{interface}
+ \text{verification}
+ \text{status}
$$

| Requirement | Question |
| --- | --- |
| Goal | 무엇을 만들거나 운영하려는가? |
| Artifact | 코드, 문서 묶음, pipeline, report, workflow 중 무엇이 남는가? |
| Interface | 누가 어떻게 사용하거나 재실행하는가? |
| Verification | 성공을 어떤 test, build, report, public evidence로 확인하는가? |
| Status | active, draft, archived 중 어디인가? |

이 다섯 항목 중 artifact와 verification이 없으면 먼저 [[research/index|Research]], [[posts/index|Posts]], [[papers/index|Papers]], 또는 [[concepts/index|Concepts]]에 둡니다.

## Promotion Flow

Project page는 보통 아래 흐름으로 생깁니다.

$$
\text{question}
\rightarrow
\text{prototype}
\rightarrow
\text{artifact}
\rightarrow
\text{project note}
\rightarrow
\text{release or post}
$$

| Starting point | Promote to project when |
| --- | --- |
| Research note | a concrete experiment, pipeline, or tool exists |
| Infra runbook | reusable utility, dashboard, or automation exists |
| Agent workflow | repeatable workflow has inputs, tools, outputs, and verification |
| Paper review | implementation or reproduction plan becomes active |
| Blog post | post describes a maintained artifact or workflow |

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

## Project Note Quality Gate

| Gate | Pass condition |
| --- | --- |
| Public-safe | internal path, account, server, collaborator, unpublished result removed |
| Reproducible enough | reader can understand inputs, outputs, and verification without private context |
| Linked | relevant concept, paper, infra, and post pages are connected |
| Scoped | project does not claim broader research results than it verifies |
| Current | status and next public action are explicit |

## 형식

| 필요 | 형식 |
| --- | --- |
| Project state | [Project lifecycle](/projects/project-lifecycle), [Project note format](/projects/project-note-format) |
| Milestone or release | [Project milestone format](/projects/project-milestone-format), [Project artifact release](/projects/project-artifact-release) |
| Model or interface | [Model card](/concepts/systems/model-card), [Inference contract](/concepts/systems/inference-contract) |
| Research decision | [Decision record](/concepts/research-methodology/decision-record), [Experiment ledger](/concepts/research-methodology/experiment-ledger) |
| Paper reproduction | [Implementation readiness](/papers/reproducibility/implementation-readiness), [Reproduction result](/papers/reproducibility/reproduction-result) |

## Candidate Projects

아래 항목은 구현물이나 workflow로 구체화할 수 있는 방향입니다. 질문만 남아 있으면 [[research/index|Research]]에 두고, 반복해서 쓸 수 있는 artifact가 생기면 project note로 확장합니다.

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
