---
title: Paper Brief Agent Pipeline
tags:
  - projects
  - agents
  - papers
---

# Paper Brief Agent Pipeline

The paper brief agent pipeline is a public-facing design for turning daily paper discovery into curated wiki notes. Raw candidates are not treated as finished knowledge.

## Problem

Daily paper lists are useful but noisy. The durable output should be a small set of verified notes, concept updates, and synthesis posts.

## Pipeline

1. Collect public paper candidates.
2. Keep raw candidates in [[inbox/index|Inbox]].
3. Promote selected papers into [[papers/index|Papers]].
4. Extract reusable ideas into [[concepts/index|Concepts]].
5. Link research relevance into [[research/index|Research]].
6. Write Korean synthesis in [[posts/index|Posts]] when a theme becomes clear.

## Acceptance Criteria

- Paper metadata is verified before publication.
- Claims are separated from interpretation.
- Missing details are marked as unresolved, not guessed.
- Agent output is reviewed before becoming a public note.

## Related

- [[agents/paper-brief-workflow|Paper brief workflow]]
- [[agents/verification-loop|Verification loop]]
- [[papers/paper-note-format|Paper note format]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
