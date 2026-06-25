---
title: Reading Status
tags:
  - papers
  - workflows
---

# Reading Status

Reading status keeps paper notes honest. It separates raw candidates from verified notes and prevents rough agent output from looking like a finished review.

## Status Values

- `candidate`: found by search or agent brief; not yet reviewed.
- `triaged`: relevant enough to keep; metadata still may need checking.
- `reading`: currently being read; claims are incomplete.
- `verified`: metadata, method, evaluation, and limits have been checked.
- `synthesis`: mature enough to support a post or research map.
- `archived`: not central, superseded, or kept only as context.

## Promotion Rules

- Candidate to triaged: topic relevance is clear and [[papers/paper-triage|Paper triage]] has a destination.
- Triaged to reading: source link and basic metadata are present.
- Reading to verified: claims are tied to evidence, metrics, and limitations.
- Verified to synthesis: the paper changes how a concept, method, or workflow is explained.

## Checks

- Does the note show what is verified and what is still uncertain?
- Are agent-generated summaries reviewed before promotion?
- Is the status consistent with the amount of evidence in the note?
- Are unpublished results or private project context excluded?

## Related

- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/paper-triage|Paper triage]]
- [[papers/paper-note-format|Paper note format]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
