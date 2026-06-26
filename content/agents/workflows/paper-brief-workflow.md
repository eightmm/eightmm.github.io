---
title: Paper Brief Workflow
tags:
  - agents
  - papers
  - workflows
---

# Paper Brief Workflow

Paper discovery agents can collect candidate papers, but the public wiki should not treat raw candidates as finished reviews. The useful workflow is ingestion, curation, linking, and synthesis.

## Roles

- Discovery agent: collects candidate papers and creates a daily brief.
- Wiki editor: converts the brief into sanitized Quartz notes.
- Human reviewer: decides what becomes a curated paper note or public post.

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

## Rules

- Do not invent DOI, arXiv IDs, metrics, datasets, or claims.
- Every candidate needs source, metadata, route, main axis, candidate claim, evidence pointer, risk, next action, and status.
- Mark missing details as `to verify`.
- Mark missing code, data, split, config, weight, log, prediction, and environment artifacts as `to verify` instead of assuming they exist.
- Prefer concept growth over paper log accumulation.
- Keep raw or uncertain entries out of polished posts.

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
