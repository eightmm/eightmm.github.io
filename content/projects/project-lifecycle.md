---
title: Project Lifecycle
tags:
  - projects
  - workflows
---

# Project Lifecycle

Project lifecycle은 idea가 public artifact가 되고, 그 artifact가 검증되며, 언제 post나 research page에서 링크할 만큼 성숙했는지를 설명합니다.

Lifecycle은 아래 흐름으로 봅니다.

$$
\text{idea}
\rightarrow
\text{draft}
\rightarrow
\text{artifact}
\rightarrow
\text{verified}
\rightarrow
\text{maintained}
\rightarrow
\text{archived}
$$

모든 project가 모든 stage에 도달할 필요는 없습니다. Boundary와 missing evidence가 명확하다면 유용한 public project page는 `draft` 상태로도 남을 수 있습니다.

## Stages

| Stage | 의미 | 필요한 evidence |
| --- | --- | --- |
| `idea` | 추적할 가치가 있는 problem | public problem statement |
| `draft` | 형태가 명확함 | artifact boundary와 design sketch |
| `artifact` | 실제 산출물이 있음 | interface, workflow, reproducible output |
| `verified` | claim이 확인됨 | build, test, benchmark, review, run artifact |
| `maintained` | 시간이 지나도 유용함 | versioning, changelog, known limit |
| `archived` | reference로 보관 | reason, final status, replacement |

## Promotion Rule

Project는 아래 조건을 만족할 때 note에서 visible project로 승격합니다.

$$
\operatorname{promote}(P)
=
\operatorname{public\_safe}(P)
\land
\operatorname{artifact\_clear}(P)
\land
\operatorname{verification\_stated}(P)
$$

여기서 $P$는 project page입니다.

## Checks

- private context 없이 이해되는 public problem이 있는가?
- artifact가 tool, workflow, model, dataset, runbook, note system 중 무엇인가?
- interface가 다른 사람이 이해할 만큼 명확한가?
- verification result와 future plan이 분리되어 있는가?
- internal path, hostname, account name, collaborator detail, unpublished metric을 생략했는가?
- next public improvement가 구체적인가?

## Related

- [[projects/index|Projects]]
- [[projects/project-note-format|Project note format]]
- [[projects/project-milestone-format|Project milestone format]]
- [[projects/project-artifact-release|Project artifact release]]
- [[concepts/research-methodology/decision-record|Decision record]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
