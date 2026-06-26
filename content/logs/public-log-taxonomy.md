---
title: Public Log Taxonomy
tags:
  - logs
  - workflows
  - publishing
---

# Public Log Taxonomy

A public log taxonomy separates reusable work records from raw diaries. The goal is to decide whether a cleaned note should remain a log, become a project milestone, update a concept, or become a Korean post.

The basic classification is:

$$
\operatorname{type}(l)
\in
\{\text{incident}, \text{experiment}, \text{decision}, \text{milestone}, \text{reading}, \text{operations}\}
$$

where $l$ is a cleaned log candidate.

## Log Types

| Type | Use when | Better destination if mature |
| --- | --- | --- |
| Incident | A failure or debugging episode taught a reusable lesson | [Public incident note](/logs/public-incident-note) or [Infra](/infra) |
| Experiment | A run tested a question or hypothesis | [Research log](/concepts/research-methodology/research-log) or [Projects](/projects) |
| Decision | A design or research direction changed | [Decision record](/concepts/research-methodology/decision-record) |
| Milestone | A public project changed state | [Project milestone format](/projects/project-milestone-format) |
| Reading | Paper reading produced a reusable insight | [Papers](/papers) or [Concepts](/concepts) |
| Operations | A workflow or runbook improved | [Infra](/infra) or [Agent workflows](/agents/workflows) |

## Minimum Fields

- Context: generic public context.
- Trigger: what caused the log to be written.
- Evidence: command, observation, source, or artifact summary.
- Lesson: reusable diagnosis, decision, or pattern.
- Destination: stay as log, promote, merge, or drop.
- Boundary: what private detail was omitted.

## Checks

- Is the log a reusable lesson rather than a private diary entry?
- Does it have enough evidence to support the lesson?
- Is the strongest destination a log, concept, project, paper, infra note, or post?
- Are unpublished results, private infrastructure, and collaborator context removed?
- Is there a next promotion rule if the log becomes important?

## Related

- [[logs/index|Public logs]]
- [[logs/public-log-format|Public log format]]
- [[logs/log-promotion-rule|Log promotion rule]]
- [[inbox/inbox-triage|Inbox triage]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
