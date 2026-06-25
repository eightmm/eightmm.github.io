---
title: Paper Brief Agent Pipeline
tags:
  - projects
  - agents
  - papers
---

# Paper Brief Agent Pipeline

The paper brief agent pipeline is a public-facing design for turning daily paper discovery into curated wiki notes. Raw candidates are not treated as finished knowledge.

## Artifact

The artifact is a curation workflow: paper candidates enter the inbox, selected items become paper notes, and reusable ideas are extracted into concept notes.

## Problem

Daily paper lists are useful but noisy. The durable output should be a small set of verified notes, concept updates, and synthesis posts.

## Public Boundary

The workflow may mention public paper metadata and public interpretation. It must not include private project priorities, collaborator requests, internal task names, or unpublished reproduction results.

## Pipeline

1. Collect public paper candidates.
2. Keep raw candidates in [[inbox/index|Inbox]].
3. Promote selected papers into [[papers/index|Papers]].
4. Extract reusable ideas into [[concepts/index|Concepts]].
5. Link research relevance into [[research/index|Research]].
6. Write Korean synthesis in [[posts/index|Posts]] when a theme becomes clear.

## Artifact Release

- Daily candidate list: inbox or private draft until sanitized.
- Curated paper note: released after metadata and claims are checked.
- Reproduction plan: released only when based on public artifacts.
- Reproduction result: released only as public-safe evidence, never as unpublished sensitive result.

## Acceptance Criteria

- Paper metadata is verified before publication.
- Claims are separated from interpretation.
- Missing details are marked as unresolved, not guessed.
- Agent output is reviewed before becoming a public note.
- Artifact release status is explicit for briefs, curated notes, and reproduction outputs.

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-artifact-release|Project artifact release]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[inbox/inbox-triage|Inbox triage]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[papers/paper-note-format|Paper note format]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/reading-status|Reading status]]
- [[projects/project-note-format|Project note format]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
