---
title: Inbox
tags:
  - inbox
  - workflows
---

# Inbox

Inbox note는 daily candidate가 curated paper note, concept note, research note, post가 되기 전에 임시로 머무는 public-safe 공간입니다.

## 용도

- daily paper brief.
- 아직 읽지 않은 candidate paper.
- 짧은 follow-up queue.
- public synthesis 전에 검증이 필요한 note.

## Workflow

- [[inbox/paper-candidate-intake|Paper candidate intake]]
- [[inbox/inbox-triage|Inbox triage]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[logs/sanitization-checklist|Sanitization checklist]]

## 상태 label

- `inbox`: 모았지만 아직 curate하지 않음.
- `stub`: 최소 공개 note.
- `reading`: 더 읽기로 선택함.
- `draft`: post 또는 synthesis note로 다듬는 중.

## Daily Brief Template

```markdown
---
title: Daily Paper Brief - YYYY-MM-DD
date: YYYY-MM-DD
tags:
  - daily-paper-brief
status: inbox
source: to verify
---

# Daily Paper Brief - YYYY-MM-DD

## Summary

- Source: to verify
- Scope: to verify
- Notable signals: to verify

## Candidates

### Paper title

- Source: to verify
- Metadata: authors `to verify`; year `to verify`; venue/status `to verify`
- Why collected: to verify
- Primary route: paper / concept update / benchmark note / post idea / drop
- Main axis: architecture / learning method / generative model / molecular modeling / Math / evaluation / systems / agents
- Candidate claim: to verify
- Evidence pointer: to verify
- Related wiki notes:
  - [Coverage matrix](/concepts/coverage-matrix)
- Risk: to verify
- Next action: verify metadata
- Status: inbox
```

## Related

- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/paper-candidate-intake|Paper candidate intake]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/index|Papers]]
- [[concepts/index|Concepts]]
- [[posts/index|Posts]]
