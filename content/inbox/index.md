---
title: Inbox
tags:
  - inbox
  - workflows
---

# Inbox

Inbox notes hold sanitized daily candidates before they become curated paper notes, concept notes, research notes, or posts.

## Use For

- Daily paper briefs.
- Candidate papers not yet read.
- Short follow-up queues.
- Notes that need verification before public synthesis.

## Workflow

- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[logs/sanitization-checklist|Sanitization checklist]]

## Status Labels

- `inbox`: collected but not curated.
- `stub`: minimal public note.
- `reading`: selected for closer reading.
- `draft`: being shaped into a post or synthesis note.

## Daily Brief Template

```markdown
---
title: Daily Paper Brief - YYYY-MM-DD
date: YYYY-MM-DD
tags:
  - daily-paper-brief
status: inbox
source: to verify
---

# Daily Paper Brief - YYYY-MM-DD

## Summary

- Source: to verify
- Scope: to verify
- Notable signals: to verify

## Candidates

### Paper title

- Source metadata: to verify
- Track: [[research/index|Research]]
- Related:
  - [[concepts/index|Concepts]]
- Status: inbox
- Follow-up:
  - Verify metadata.
  - Check relevance.
  - Check benchmark and leakage issues.
```

## Related

- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/index|Papers]]
- [[concepts/index|Concepts]]
- [[posts/index|Posts]]
