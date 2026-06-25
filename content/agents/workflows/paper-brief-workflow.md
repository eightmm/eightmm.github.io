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
2. Unclear items stay in [[inbox/curation-queue|Curation queue]].
3. Interesting items pass [[papers/paper-triage|Paper triage]].
4. Selected items become [[papers/index|Paper]] stubs with [[papers/reading-status|reading status]].
5. Reusable ideas update [[concepts/index|Concepts]].
6. Research relevance is linked into [[research/index|Research]].
7. Public promotion passes [[inbox/publishing-gate|Publishing gate]].
8. Weekly or monthly synthesis becomes [[posts/index|Posts]].

## Rules

- Do not invent DOI, arXiv IDs, metrics, datasets, or claims.
- Mark missing details as `to verify`.
- Prefer concept growth over paper log accumulation.
- Keep raw or uncertain entries out of polished posts.

## Related

- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[papers/paper-note-format|Paper note format]]
- [[papers/paper-triage|Paper triage]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/claim-extraction|Claim extraction]]
- [[papers/evidence-table|Evidence table]]
- [[papers/reproducibility-checklist|Reproducibility checklist]]
- [[papers/index|Papers]]
- [[concepts/index|Concepts]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
