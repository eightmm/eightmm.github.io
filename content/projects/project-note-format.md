---
title: Project Note Format
tags:
  - projects
  - workflows
---

# Project Note Format

A project note explains a public artifact or workflow as a reusable engineering story. It should be useful without private repository paths, server details, internal task names, or unpublished experimental results.

## Role

Use a project note when the page is about something being built or operated:

- A research pipeline.
- A public tool or template.
- An agent workflow.
- A reproducible infrastructure pattern.
- A blog/wiki maintenance system.

Use a concept note instead when the page is mainly a reusable definition. Use a paper note when the page is mainly about one paper.

## Minimal Model

$$
P = (q, a, d, m, v, b)
$$

where:

- $q$ is the public problem or question.
- $a$ is the artifact: code, workflow, note system, pipeline, or runbook.
- $d$ is the public data or input boundary.
- $m$ is the method or design.
- $v$ is the verification evidence.
- $b$ is the public boundary: what must not be exposed.

## Suggested Sections

### Problem

State the problem in a way that does not depend on private context.

### Artifact

Describe what exists or what is being built: a pipeline, note system, runbook, agent workflow, or tool.

### Public Boundary

State what the note intentionally omits:

- Private paths, hostnames, account names, SSH details, credentials, and user lists.
- Internal project names or collaborator-specific context.
- Private datasets, unpublished metrics, and thesis-sensitive results.

### Design

Explain the interfaces, constraints, and design decisions. Link to concepts rather than repeating full definitions.

### Verification

Record public checks:

- Build or syntax check.
- Unit, integration, or smoke test.
- Manual review checklist.
- Reproducibility check.
- Known limitation.

### Status

Use a small status vocabulary:

- `idea`: useful direction, not implemented.
- `draft`: structure exists but needs evidence.
- `active`: being used or iterated.
- `paused`: useful but not current.
- `archived`: kept for reference.

### Next Work

List the next public improvement, not a private task tracker.

## Template

```markdown
---
title: Project Name
tags:
  - projects
status: draft
---

# Project Name

## Problem

Public problem statement.

## Artifact

What exists or is being built.

## Public Boundary

What is intentionally omitted.

## Design

Key interfaces and decisions.

## Verification

- Public check.

## Status

Current state.

## Next Work

- Next public improvement.

## Related

- [[projects/index|Projects]]
```

## Promotion Rule

A project note is ready for public linking when:

- The problem is understandable without private context.
- The artifact has a clear interface or workflow.
- Verification is stated separately from aspiration.
- Risks and missing evidence are explicit.
- Related concept, paper, infra, or agent notes are linked.

## Related

- [[projects/index|Projects]]
- [[projects/project-milestone-format|Project milestone format]]
- [[concepts/research-methodology/decision-record|Decision record]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[inbox/publishing-gate|Publishing gate]]
