---
title: Curation Queue
tags:
  - inbox
  - workflows
  - curation
---

# Curation Queue

A curation queue holds items that are promising but not yet ready to become polished wiki notes, paper notes, project notes, logs, or posts. It keeps the public site from filling with raw fragments while preserving follow-up work.

Each queued item should have:

$$
q_i
=
(\text{source}, \text{destination}, \text{status}, \text{blocker}, \text{next action})
$$

## Queue Fields

- Source: where the item came from, marked `to verify` if uncertain.
- Destination: paper, concept, project, research, log, or post.
- Promotion workflow: target destination and reason, following [[agents/workflows/content-promotion-workflow|Content promotion workflow]].
- Status: `inbox`, `stub`, `reading`, `draft`, `ready`, or `drop`.
- Blocker: missing metadata, weak relevance, private detail, missing verification, or duplicate.
- Next action: the smallest useful action.

## Promotion Rules

- Promote to [[papers/index|Papers]] when the item is paper-specific, metadata is verified, and it passes [[papers/paper-triage|Paper triage]].
- Before reproduction planning, record paper-specific public artifacts with [[papers/artifact-availability|Artifact availability]].
- Promote to [[concepts/index|Concepts]] when the idea is reusable across papers or projects.
- Promote to [[projects/index|Projects]] when it describes a public artifact or design decision.
- Promote to [[logs/index|Public logs]] when it is a cleaned work record.
- Promote to [[posts/index|Posts]] only when there is a reader-facing story or synthesis.

## Drop Rules

- Drop duplicates when an existing note only needs an update.
- Drop unverifiable items that cannot be sourced.
- Drop or keep private anything that cannot be sanitized.
- Drop items that are merely interesting but do not fit the public scope.

## Checks

- Is the next destination explicit?
- Is the smallest next action concrete?
- Has sensitive information been removed before public promotion?
- Is `to verify` used for missing metadata instead of invented details?
- Is the queue shrinking through decisions rather than growing forever?

## Related

- [[inbox/index|Inbox]]
- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/publishing-gate|Publishing gate]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[papers/reading-status|Reading status]]
- [[papers/paper-triage|Paper triage]]
- [[papers/artifact-availability|Artifact availability]]
