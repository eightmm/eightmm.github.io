---
title: Projects
tags:
  - projects
---

# Projects

Project pages describe goals, interfaces, design decisions, verification methods, and public artifacts. 이 영역은 아이디어 자체보다 실제로 만들거나 운영하거나 공개 산출물로 남기는 것에 둡니다.

## Role

Use this area for artifacts and workflows that are being built, operated, or iterated. A project page should connect a problem to an artifact, design decision, verification method, current status, and next public improvement.

If the page is only a reusable definition, put it under [[concepts/index|Concepts]]. If it is centered on one paper, put it under [[papers/index|Papers]]. If it is an operational lesson without a project artifact, put it under [[infra/index|Infra]] or [[logs/index|Logs]]. If it is mostly a question, hypothesis, or research direction, put it under [[research/index|Research]] first.

## Research와의 차이

| Page Center | Put In |
| --- | --- |
| public question, hypothesis, research idea | [[research/index|Research]] |
| implemented pipeline, tool, artifact, workflow | [[projects/index|Projects]] |
| paper-specific claim | [[papers/index|Papers]] |
| reusable concept or formula | [[concepts/index|Concepts]] |
| operational lesson without an artifact | [[infra/index|Infra]] or [[logs/index|Logs]] |

## Buckets

| Bucket | Use For |
| --- | --- |
| Research systems | pipelines, experiment workflows, reproducible artifacts |
| Structure-based modeling | docking, scoring, pose quality, screening workflows |
| Protein modeling | representation, structure, sequence-structure tools |
| HPC / server utilities | Slurm, GPU, storage, environment workflow tools |
| Knowledge-base tooling | blog/wiki structure, paper curation, note promotion |
| Agent workflows | coding, paper briefs, verification, orchestration |

## Project Notes

| Project | Focus |
| --- | --- |
| [LLM Wiki blog](/projects/llm-wiki-blog) | public knowledge-base structure |
| [Paper brief agent pipeline](/projects/paper-brief-agent-pipeline) | paper discovery and curation workflow |
| [HPC research workflows](/projects/hpc-research-workflows) | reproducible research engineering on shared compute |

## Formats

| Need | Format |
| --- | --- |
| Project state | [Project lifecycle](/projects/project-lifecycle), [Project note format](/projects/project-note-format) |
| Milestone or release | [Project milestone format](/projects/project-milestone-format), [Project artifact release](/projects/project-artifact-release) |
| Model or interface | [Model card](/concepts/systems/model-card), [Inference contract](/concepts/systems/inference-contract) |
| Research decision | [Decision record](/concepts/research-methodology/decision-record), [Experiment ledger](/concepts/research-methodology/experiment-ledger) |
| Paper reproduction | [Implementation readiness](/papers/reproducibility/implementation-readiness), [Reproduction result](/papers/reproducibility/reproduction-result) |

## Future Candidates

| Candidate | Direction |
| --- | --- |
| Structure-based screening pipeline | connect SBDD workflow, scoring, and evaluation |
| Protein-modeling experiment template | standardize representation, split, metric, and run records |
| Slurm/GPU monitoring utility | public runbook and dashboard design notes |
| Agent workflow notes | research engineering and verification workflows |

## Related

| Area | Link |
| --- | --- |
| Research | [Research](/research), [Research methodology](/concepts/research-methodology) |
| Systems | [Run artifact](/concepts/systems/run-artifact), [Slurm](/infra/hpc/slurm) |
| Papers | [Papers](/papers) |
| Logs and inbox | [Public logs](/logs), [Public log format](/logs/public-log-format), [Publishing gate](/inbox/publishing-gate) |
| Agents | [Agents](/agents) |
