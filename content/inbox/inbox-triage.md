---
title: Inbox Triage
tags:
  - inbox
  - workflows
---

# Inbox Triage

Inbox triage decides whether a raw candidate becomes a paper note, concept note, project note, post, public log, or deletion. It prevents the wiki from filling with unverified fragments.

## Triage Questions

- Is the source public and verifiable?
- Is the note a reusable concept, a paper-specific claim, a project record, or a blog idea?
- Does it contain private or unpublished information that must be removed?
- Does it need human review before publication?
- What is the smallest useful destination page?
- If it is a paper candidate, does it pass [[papers/paper-triage|Paper triage]]?
- Which destination has the strongest fit according to [[agents/workflows/content-promotion-workflow|Content promotion workflow]]?

## Destination Rules

- Paper-specific claim: [[papers/index|Papers]]
- Reusable term or method: [[concepts/index|Concepts]]
- Research direction: [[research/index|Research]]
- Public implementation or workflow: [[projects/index|Projects]]
- Clean work record: [[logs/index|Public logs]]
- Korean narrative: [[posts/index|Posts]]

If the destination is unclear, keep the item in [[inbox/curation-queue|Curation queue]] with a concrete next action rather than creating a weak page.

## Checks

- Do not promote raw agent output without verification.
- Do not create a new page when an existing page only needs a short update.
- Mark unresolved metadata as `to verify`.
- Pass [[inbox/publishing-gate|Publishing gate]] before public promotion.
- Delete or keep private anything that cannot be safely generalized.
- Prefer concept updates over creating many weak paper stubs.

## Related

- [[inbox/index|Inbox]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[papers/reading-status|Reading status]]
- [[papers/paper-triage|Paper triage]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[logs/sanitization-checklist|Sanitization checklist]]
