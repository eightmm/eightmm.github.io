---
title: Paper Brief Agent Pipeline
tags:
  - projects
  - agents
  - papers
---

# Paper Brief Agent Pipeline

Paper brief agent pipeline은 daily paper discovery를 curated wiki note로 바꾸기 위한 public-facing design입니다. Raw candidate는 finished knowledge로 취급하지 않습니다.

## Artifact

Artifact는 curation workflow입니다. Paper candidate는 inbox로 들어오고, 선택된 item은 paper note가 되며, reusable idea는 concept note로 추출됩니다.

## Status

Lifecycle status: `active`.

This project describes the public curation workflow. Individual daily briefs remain inbox material until metadata, relevance, and public boundary are checked.

## Problem

Daily paper list는 유용하지만 noisy합니다. Durable output은 검증된 note, concept update, synthesis post의 작은 묶음이어야 합니다.

## Public Boundary

Workflow는 public paper metadata와 public interpretation을 다룰 수 있습니다. Private project priority, collaborator request, internal task name, unpublished reproduction result는 포함하지 않습니다.

## Pipeline

1. public paper candidate를 수집합니다.
2. raw candidate를 [[inbox/index|Inbox]]에 둡니다.
3. 선택된 paper를 [[papers/index|Papers]]로 promote합니다.
4. reusable idea를 [[concepts/index|Concepts]]로 추출합니다.
5. research relevance를 [[research/index|Research]]에 연결합니다.
6. theme이 명확해지면 [[posts/index|Posts]]에 Korean synthesis를 씁니다.

## Artifact Release

- Daily candidate list: sanitized되기 전까지 inbox 또는 private draft.
- Curated paper note: metadata와 claim을 확인한 뒤 released.
- Reproduction plan: public artifact에 기반할 때만 released.
- Reproduction result: unpublished sensitive result가 아니라 public-safe evidence로만 released.

| Artifact | Release status |
| --- | --- |
| Workflow description | released |
| Curated paper note | released after review |
| Raw daily candidate list | inbox or not released until sanitized |
| Agent draft and private priority | not released |
| Reproduction result | released only when public-safe and evidence-backed |

## Acceptance Criteria

- publication 전에 paper metadata가 verified되어야 합니다.
- claim과 interpretation을 분리해야 합니다.
- missing detail은 guessed가 아니라 unresolved로 표시해야 합니다.
- agent output은 public note가 되기 전에 review되어야 합니다.
- brief, curated note, reproduction output의 artifact release status가 explicit해야 합니다.

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-artifact-release|Project artifact release]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[inbox/inbox-triage|Inbox triage]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/reading-status|Reading status]]
- [[projects/project-note-format|Project note format]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
