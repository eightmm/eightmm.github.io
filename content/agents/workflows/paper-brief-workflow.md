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

1. Daily brief enters [[inbox/index|Inbox]].
2. Each candidate uses [[inbox/paper-candidate-intake|Paper candidate intake]].
3. Unclear items stay in [[inbox/curation-queue|Curation queue]].
4. Interesting items pass [[papers/workflows/paper-triage|Paper triage]].
5. Selected items become [[papers/index|Paper]] stubs with [[papers/workflows/reading-status|reading status]].
6. Public materials are recorded with [[papers/reproducibility/artifact-availability|Artifact availability]].
7. Implementation candidates pass [[papers/reproducibility/implementation-readiness|Implementation readiness]].
8. Reruns or diagnostics get a [[papers/reproducibility/reproduction-plan|Reproduction plan]] and [[papers/reproducibility/reproduction-result|Reproduction result]].
9. Reusable ideas update [[concepts/index|Concepts]] through [[papers/workflows/concept-update-contract|Concept update contract]].
10. Research relevance is linked into [[research/index|Research]].
11. Public promotion passes [[inbox/publishing-gate|Publishing gate]].
12. Weekly or monthly synthesis becomes [[posts/index|Posts]].

## 규칙

- DOI, arXiv ID, metric, dataset, claim을 지어내지 않습니다.
- 모든 candidate에는 source, metadata, route, main axis, candidate claim, evidence pointer, risk, next action, status가 필요합니다.
- missing detail은 `to verify`로 표시합니다.
- code, data, split, config, weight, log, prediction, environment artifact가 없으면 있다고 가정하지 말고 `to verify`로 표시합니다.
- paper log를 쌓기보다 concept growth를 우선합니다.
- raw 또는 uncertain entry는 polished post에 넣지 않습니다.

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
