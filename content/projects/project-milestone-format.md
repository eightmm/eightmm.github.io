---
title: Project Milestone Format
tags:
  - projects
  - workflows
---

# Project Milestone Format

A project milestone records a public change in a project without exposing private task names, unpublished results, or internal infrastructure.

## Suggested Shape

- Goal: what the milestone tried to make possible.
- Change: what changed at a high level.
- Verification: what was checked.
- Risk: what remains uncertain.
- Next step: what would make the project more useful.

## Minimal Template

```markdown
---
title: Project Milestone - YYYY-MM-DD - Topic
date: YYYY-MM-DD
tags:
  - projects
status: milestone
---

# Project Milestone - Topic

## Goal

Public goal.

## Change

Public change summary.

## Verification

- Build, test, review, or manual check.

## Related

- [[projects/index|Projects]]
```

## Checks

- Is the milestone useful without private details?
- Does it link to concepts, papers, infra, or agent workflows?
- Are verification results public and reproducible?
- Are unpublished metrics omitted or generalized?

## Related

- [[projects/index|Projects]]
- [[logs/public-log-format|Public log format]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[agents/verification-loop|Verification loop]]
