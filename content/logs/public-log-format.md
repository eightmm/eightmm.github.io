---
title: Public Log Format
tags:
  - logs
  - workflows
---

# Public Log Format

A public log is a cleaned record of work that is safe to publish. It should preserve the lesson, diagnosis, and reusable method without exposing private infrastructure, collaborators, datasets, or unpublished results.

Use [[logs/public-log-taxonomy|Public log taxonomy]] to classify the log and [[logs/log-promotion-rule|Log promotion rule]] to decide whether it should remain a log or move elsewhere.

## Suggested Shape

- Context: what kind of work was happening.
- Symptom: what failed or what changed.
- Diagnosis: how the issue was narrowed down.
- Resolution: what fixed it or what decision was made.
- General lesson: what can be reused later.
- Links: related project, infra, concept, or paper notes.
- Destination: stay as log, merge into a note, promote to milestone, or become a post candidate.

For operational failures, use [[logs/public-incident-note|Public incident note]].

## Minimal Template

```markdown
---
title: Public Log - YYYY-MM-DD - Topic
date: YYYY-MM-DD
tags:
  - logs
status: public-log
---

# Public Log - Topic

## Context

Generic, public context only.

## What Changed

- Public summary.

## What Was Learned

- Reusable lesson.

## Related

- [[projects/index|Projects]]
```

## Checks

- Does the log teach a reusable pattern rather than expose private operations?
- Are exact paths, accounts, endpoints, ports, and node names removed?
- Are unpublished metrics or internal task names absent?
- Are related notes linked so the log is not an isolated diary entry?
- Is the promotion destination explicit?

## Related

- [[logs/index|Public logs]]
- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/log-promotion-rule|Log promotion rule]]
- [[logs/public-incident-note|Public incident note]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[projects/project-milestone-format|Project milestone format]]
