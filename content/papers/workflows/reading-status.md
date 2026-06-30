---
title: Reading Status
unlisted: true
aliases:
  - papers/reading-status
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

## Evidence Required

| Status | Required evidence |
| --- | --- |
| `candidate` | source link or identifier |
| `triaged` | route, topic relevance, and reason to keep |
| `reading` | metadata checked and open questions listed |
| `verified` | claims tied to evidence table, benchmark card, and limitations |
| `synthesis` | concept updates or post/project links exist |
| `archived` | reason for archive is stated |

Status is not a progress label for effort spent. It is a claim about evidence quality.

## Promotion Rules

- Candidate to triaged: topic relevance is clear and [[papers/workflows/paper-triage|Paper triage]] has a destination.
- Triaged to reading: source link and basic metadata are present.
- Reading to verified: claims are tied to evidence, metrics, and limitations.
- Verified to synthesis: the paper changes how a concept, method, or workflow is explained.

## Demotion Rules

Demote a note when later inspection weakens the evidence:

| From | To | Trigger |
| --- | --- | --- |
| verified | reading | key metric or split was misread |
| synthesis | verified | synthesis claim is broader than evidence |
| triaged | archived | off-scope or superseded |
| reading | candidate | metadata/source cannot be verified |

Demotion is not failure; it keeps public notes honest.

## Checks

- Does the note show what is verified and what is still uncertain?
- Are agent-generated summaries reviewed before promotion?
- Is the status consistent with the amount of evidence in the note?
- Are unpublished results or private project context excluded?
- Would a reader know which claims are safe to reuse?
- Is `to verify` used instead of invented metadata or metrics?

## Related

- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
