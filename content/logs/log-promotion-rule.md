---
title: Log Promotion Rule
tags:
  - logs
  - publishing
  - workflows
---

# Log Promotion Rule

A log promotion rule decides when a cleaned log should remain a log, merge into an existing note, become a project milestone, or seed a Korean post.

For a log $l$, choose the destination:

$$
D(l)
=
\arg\max_{d \in \mathcal{D}}
\operatorname{fit}(l, d)
$$

where $\mathcal{D}$ includes log, concept, paper, project, infra, agent workflow, research note, and post.

## Promotion Targets

| Promote to | When |
| --- | --- |
| Existing concept | The log clarifies a reusable definition, equation, metric, or checklist |
| Existing infra note | The lesson is a general operating pattern |
| Project milestone | The log records a public project state change |
| Paper note | The log is tied to one paper claim or reproduction result |
| Research methodology | The log changes experiment design, hypothesis, result interpretation, or threat model |
| Korean post | Multiple logs and wiki notes form a reader-facing story |
| Drop or keep private | The note cannot be verified or sanitized |

## Decision Function

Promotion should require:

$$
\operatorname{promote}(l)
=
\operatorname{safe}(l)
\land
\operatorname{evidence}(l)
\land
\operatorname{reuse}(l)
\land
\operatorname{linked}(l)
$$

The log should stay in `logs/` when it is useful but not yet broad enough to become a canonical note.

## Checks

- Would merging this into an existing note reduce duplication?
- Does the log have evidence, or is it only a memory?
- Is the public lesson independent of private context?
- Is the next destination linked from the log?
- Does promotion require human review, paper metadata verification, or artifact release review?

## Related

- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/public-log-format|Public log format]]
- [[inbox/publishing-gate|Publishing gate]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[projects/project-milestone-format|Project milestone format]]
- [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
