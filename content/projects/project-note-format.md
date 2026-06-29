---
title: Project Note Format
tags:
  - projects
  - workflows
---

# Project Note Format

Project note는 public artifact나 workflow를 재사용 가능한 engineering story로 설명합니다. Private repository path, server detail, internal task name, unpublished experimental result 없이도 유용해야 합니다.

## 역할

Page가 만들고 있거나 운영 중인 것을 다룰 때 project note를 씁니다.

- research pipeline.
- public tool 또는 template.
- agent workflow.
- reproducible infrastructure pattern.
- blog/wiki maintenance system.

Page의 중심이 reusable definition이면 concept note를 씁니다. Paper 하나가 중심이면 paper note를 씁니다.

Page가 idea, draft, artifact, verified project, maintained project, archive 중 어디에 있는지는 [[projects/project-lifecycle|Project lifecycle]]로 판단합니다.

## Minimal Model

$$
P = (q, a, d, m, v, b)
$$

여기서:

- $q$는 public problem 또는 question입니다.
- $a$는 code, workflow, note system, pipeline, runbook 같은 artifact입니다.
- $d$는 public data 또는 input boundary입니다.
- $m$은 method 또는 design입니다.
- $v$는 verification evidence입니다.
- $b$는 공개하면 안 되는 public boundary입니다.

## 권장 section

### Problem

Private context에 의존하지 않는 방식으로 problem을 적습니다.

### Artifact

이미 존재하는 것 또는 만들고 있는 것을 설명합니다. Pipeline, note system, runbook, agent workflow, tool이 여기에 해당합니다.

Model 또는 inference artifact라면 필요할 때 public [[concepts/systems/model-card|Model card]]와 [[concepts/systems/inference-contract|Inference contract]]를 연결합니다.

Released 또는 withheld artifact는 [[projects/project-artifact-release|Project artifact release]]를 사용합니다.

### Public Boundary

Note가 의도적으로 생략하는 것을 적습니다.

- private path, hostname, account name, SSH detail, credential, user list.
- internal project name 또는 collaborator-specific context.
- private dataset, unpublished metric, thesis-sensitive result.

### Design

Interface, constraint, design decision을 설명합니다. 전체 definition을 반복하기보다 concept로 link합니다.

### Verification

Public check를 기록합니다.

- build 또는 syntax check.
- unit, integration, smoke test.
- manual review checklist.
- reproducibility check.
- known limitation.

### Status

작은 status vocabulary를 씁니다.

- `idea`: 유용한 방향이지만 아직 구현되지 않음.
- `draft`: 구조는 있지만 evidence가 더 필요함.
- `active`: 사용 중이거나 반복 개선 중.
- `paused`: 유용하지만 현재 진행 중은 아님.
- `archived`: reference로 보관.

### Next Work

Private task tracker가 아니라 다음 public improvement를 적습니다.

## Template

```markdown
---
title: Project Name
tags:
  - projects
status: draft
---

# Project Name

## Problem

Public problem statement.

## Artifact

What exists or is being built.

## Public Boundary

What is intentionally omitted.

## Design

Key interfaces and decisions.

## Verification

- Public check.

## Status

Current state.

## Next Work

- Next public improvement.

## Related

- [[projects/index|Projects]]
```

## Promotion rule

Project note는 아래 조건을 만족할 때 public linking에 적합합니다.

- problem이 private context 없이 이해됩니다.
- artifact에 clear interface 또는 workflow가 있습니다.
- model 또는 inference artifact에는 필요할 때 clear model card 또는 inference contract가 있습니다.
- verification이 aspiration과 분리되어 적혀 있습니다.
- risk와 missing evidence가 explicit합니다.
- related concept, paper, infra, agent note가 link되어 있습니다.
- artifact release status가 `released`, `not released`, `to verify`, `not applicable`, `replaced by summary` 중 하나로 explicit합니다.

## Related

- [[projects/index|Projects]]
- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-milestone-format|Project milestone format]]
- [[projects/project-artifact-release|Project artifact release]]
- [[concepts/research-methodology/decision-record|Decision record]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[inbox/publishing-gate|Publishing gate]]
