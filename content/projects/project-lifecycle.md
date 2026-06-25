---
title: Project Lifecycle
tags:
  - projects
  - workflows
---

# Project Lifecycle

A project lifecycle describes how an idea becomes a public artifact, how the artifact is verified, and when it is mature enough to be linked from posts or research pages.

The lifecycle is:

$$
\text{idea}
\rightarrow
\text{draft}
\rightarrow
\text{artifact}
\rightarrow
\text{verified}
\rightarrow
\text{maintained}
\rightarrow
\text{archived}
$$

Not every project needs to reach every stage. A useful public project page can stay at `draft` if it clearly states its boundary and missing evidence.

## Stages

| Stage | Meaning | Required evidence |
| --- | --- | --- |
| `idea` | The problem is worth tracking | Public problem statement |
| `draft` | The shape is clear | Artifact boundary and design sketch |
| `artifact` | Something exists | Interface, workflow, or reproducible output |
| `verified` | Claims are checked | Build, test, benchmark, review, or run artifact |
| `maintained` | It is useful over time | Versioning, changelog, known limits |
| `archived` | Kept for reference | Reason, final status, replacement if any |

## Promotion Rule

A project should be promoted from note to visible project when:

$$
\operatorname{promote}(P)
=
\operatorname{public\_safe}(P)
\land
\operatorname{artifact\_clear}(P)
\land
\operatorname{verification\_stated}(P)
$$

where $P$ is the project page.

## Checks

- Is there a public problem independent of private context?
- Is the artifact a tool, workflow, model, dataset, runbook, or note system?
- Is the interface clear enough for another person to understand?
- Are verification results separated from future plans?
- Are internal paths, hostnames, account names, collaborator details, and unpublished metrics omitted?
- Is the next public improvement specific?

## Related

- [[projects/index|Projects]]
- [[projects/project-note-format|Project note format]]
- [[projects/project-milestone-format|Project milestone format]]
- [[projects/project-artifact-release|Project artifact release]]
- [[concepts/research-methodology/decision-record|Decision record]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
