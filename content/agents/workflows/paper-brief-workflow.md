---
title: Paper Brief Workflow
tags:
  - agents
  - papers
  - workflows
---

# Paper Brief Workflow

Paper discovery agent는 candidate paper를 모을 수 있지만, public wiki는 raw candidate를 finished review처럼 취급하면 안 됩니다. 유용한 workflow는 ingestion, curation, linking, synthesis입니다.

## Roles

- Discovery agent: candidate paper를 모으고 daily brief를 만듭니다.
- Wiki editor: brief를 sanitized Quartz note로 바꿉니다.
- Human reviewer: 무엇이 curated paper note 또는 public post가 될지 결정합니다.

## Flow

1. Daily brief는 [[inbox/index|Inbox]]로 들어갑니다.
2. 각 candidate는 [[inbox/paper-candidate-intake|Paper candidate intake]] 형식을 사용합니다.
3. route가 불명확한 item은 [[inbox/curation-queue|Curation queue]]에 남깁니다.
4. 흥미로운 item은 [[papers/workflows/paper-triage|Paper triage]]를 통과합니다.
5. 선택된 item은 [[papers/workflows/reading-status|reading status]]가 있는 [[papers/index|Paper]] stub이 됩니다.
6. Public material은 [[papers/reproducibility/artifact-availability|Artifact availability]]에 기록합니다.
7. Implementation candidate는 [[papers/reproducibility/implementation-readiness|Implementation readiness]]를 통과해야 합니다.
8. Rerun 또는 diagnostic은 [[papers/reproducibility/reproduction-plan|Reproduction plan]]과 [[papers/reproducibility/reproduction-result|Reproduction result]]로 남깁니다.
9. Reusable idea는 [[papers/workflows/concept-update-contract|Concept update contract]]를 통해 [[concepts/index|Concepts]]를 업데이트합니다.
10. Research relevance는 [[research/index|Research]]에 연결합니다.
11. Public promotion은 [[inbox/publishing-gate|Publishing gate]]를 통과합니다.
12. Weekly/monthly synthesis는 [[posts/index|Posts]]가 됩니다.

## Brief Schema

Daily candidate가 바로 paper note가 되면 품질이 무너집니다. 최소 schema는 아래 정도가 필요합니다.

| Field | Meaning |
| --- | --- |
| source | arXiv, journal, conference, repository, manual find |
| metadata | title, authors, date, identifier, link |
| axis | architecture, training, evaluation, comp-bio, infra, agents |
| candidate claim | paper가 주장하는 핵심 기여 |
| evidence pointer | abstract, figure, table, experiment section, artifact |
| route | ignore, inbox, paper stub, concept update, project idea |
| risk | hype, weak eval, leakage, missing artifact, off-scope |
| next action | read, verify, compare, implement, summarize |

Missing field는 채워 넣지 말고 `to verify`로 남깁니다.

## Promotion Gate

Paper candidate가 public wiki로 올라가려면 아래 gate 중 하나를 통과해야 합니다.

| Gate | Promote to |
| --- | --- |
| reusable idea exists | [[concepts/index|Concepts]] update |
| domain claim is relevant | [[papers/index|Papers]] note |
| implementation is plausible | [[projects/index|Projects]] idea or task |
| broad lesson is readable | [[posts/index|Posts]] synthesis |
| evidence is too weak | keep in [[inbox/index|Inbox]] |

이 gate는 paper 수를 늘리는 장치가 아니라, public wiki의 signal-to-noise를 유지하는 장치입니다.

## 규칙

- DOI, arXiv ID, metric, dataset, claim을 지어내지 않습니다.
- 모든 candidate에는 source, metadata, route, main axis, candidate claim, evidence pointer, risk, next action, status가 필요합니다.
- missing detail은 `to verify`로 표시합니다.
- code, data, split, config, weight, log, prediction, environment artifact가 없으면 있다고 가정하지 말고 `to verify`로 표시합니다.
- paper log를 쌓기보다 concept growth를 우선합니다.
- raw 또는 uncertain entry는 polished post에 넣지 않습니다.
- benchmark 수치보다 split, metric, baseline, artifact availability를 먼저 확인합니다.
- paper-specific result를 reusable concept page에 일반 사실처럼 쓰지 않습니다.
- agent가 만든 요약은 source를 대조하기 전까지 `draft`로 취급합니다.

## Related

- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[inbox/paper-candidate-intake|Paper candidate intake]]
- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]
- [[papers/index|Papers]]
- [[concepts/index|Concepts]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
