---
title: Incident Response for Shared Research Machines
tags:
  - infra
  - server-ops
  - reliability
---

# Incident Response for Shared Research Machines

Incident response is the process of turning a shared-machine failure into a bounded diagnosis, a safe recovery action, and a public lesson. The goal is not to publish raw operational detail; the goal is to preserve the reasoning pattern.

A useful incident note separates:

$$
I = (S, E, C, A, P, B)
$$

where $S$ is the symptom, $E$ is collected evidence, $C$ is the likely cause, $A$ is the action taken, $P$ is prevention, and $B$ is the public boundary.

## Public Runbook Shape

| Section | Public content |
| --- | --- |
| Symptom | Generic user-visible failure, such as job failure, full disk, slow IO, or GPU unavailable |
| Scope | Affected class of resource, not hostnames or live topology |
| Evidence | Sanitized logs, counters, scheduler state, or error class |
| Cause | Mechanism-level explanation, not private configuration |
| Action | General remediation pattern |
| Prevention | Monitoring, quota, checkpointing, documentation, or access boundary |
| Boundary | What was deliberately omitted from the public note |

## Severity

Severity should combine impact and urgency:

$$
\text{severity} = f(\text{affected users}, \text{data risk}, \text{compute loss}, \text{recovery time})
$$

For public writing, use qualitative labels such as `low`, `medium`, and `high` instead of exposing internal priority labels or incident IDs.

## Checks

- What symptom was observed first?
- Which evidence distinguishes resource exhaustion, software failure, storage failure, network failure, and user error?
- Can the recovery action be described without exposing private topology?
- Was any data integrity risk checked before rerunning jobs?
- Is there a prevention step: alert, quota, backup, checkpoint, run reconciliation, or access change?
- Does the note avoid hostnames, usernames, private paths, ports, dashboards, and internal incident IDs?

## Related

- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[infra/server-ops/backup-restore|Backup and restore]]
- [[infra/server-ops/access-boundary|Access boundary]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/observability|Observability]]
- [[logs/public-incident-note|Public incident note]]
