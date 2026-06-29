---
title: Project Milestone Format
tags:
  - projects
  - workflows
---

# Project Milestone Format

Project milestone은 private task name, unpublished result, internal infrastructure를 노출하지 않고 project의 공개 가능한 변화를 기록합니다.

## Suggested Shape

- Goal: milestone이 무엇을 가능하게 하려 했는지.
- Change: 큰 수준에서 무엇이 바뀌었는지.
- Verification: 무엇을 확인했는지.
- Artifact: 무엇이 공개됐고, 무엇이 private로 남았고, 어떤 release status가 바뀌었는지.
- Risk: 아직 불확실한 점이 무엇인지.
- Next step: project를 더 유용하게 만들 다음 공개 작업이 무엇인지.

## Minimal Template

```markdown
---
title: Project Milestone - YYYY-MM-DD - Topic
date: YYYY-MM-DD
tags:
  - projects
status: milestone
---

# Project Milestone - Topic

## Goal

Public goal.

## Change

Public change summary.

## Verification

- Build, test, review, or manual check.

## Related

- [[projects/index|Projects]]
```

## Checks

- private detail 없이도 milestone이 유용한가?
- concept, paper, infra, agent workflow와 연결되는가?
- verification result가 public하고 reproducible한가?
- artifact release status가 명확하고 public-safe한가?
- unpublished metric을 생략하거나 일반화했는가?

## Related

- [[projects/index|Projects]]
- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-artifact-release|Project artifact release]]
- [[logs/public-log-format|Public log format]]
- [[concepts/research-methodology/research-log|Research log]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[agents/verification/verification-loop|Verification loop]]
