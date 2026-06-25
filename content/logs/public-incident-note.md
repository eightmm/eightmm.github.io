---
title: Public Incident Note
tags:
  - logs
  - infra
  - workflows
---

# Public Incident Note

A public incident note is a cleaned writeup of an operational failure or debugging episode. It should preserve the reusable diagnosis and recovery pattern without exposing private infrastructure or security-sensitive details.

An incident can be summarized as:

$$
I
=
(\text{symptom}, \text{impact}, \text{cause}, \text{fix}, \text{prevention})
$$

## Suggested Shape

- Context: generic system class or workflow.
- Symptom: what failed from the user's perspective.
- Impact: what kind of work was blocked or degraded.
- Diagnosis: how the cause was narrowed down.
- Cause: root cause or best current explanation.
- Fix: public-safe resolution.
- Prevention: monitoring, checklist, test, or runbook update.
- Links: related infra, systems, project, or research notes.
- Promotion: whether the incident should remain a log or update an infra runbook.

## Public-Safe Examples

- "A GPU training job repeatedly failed after checkpoint restore."
- "A batch inference pipeline produced partial outputs after interruption."
- "A shared environment changed and broke a reproducible run."

Avoid exact hostnames, usernames, private paths, queue names, ports, internal ticket names, and unpublished experiment metrics.

## Checks

- Does the note teach a reusable operational pattern?
- Are infrastructure details generalized?
- Does the prevention step connect to a checklist or runbook?
- Is the issue already fixed or clearly marked as unresolved?
- Would publishing the note reveal security posture or private work?
- Should the prevention step update [[infra/server-ops/incident-response|Incident response for shared research machines]]?

## Related

- [[logs/public-log-format|Public log format]]
- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/log-promotion-rule|Log promotion rule]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/observability|Observability]]
- [[infra/index|Infra]]
