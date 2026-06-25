---
title: Observability
tags:
  - systems
  - monitoring
  - reliability
---

# Observability

Observability is the ability to understand a system's state from logs, metrics, traces, artifacts, and alerts. In research systems, observability helps explain why a run was slow, failed, diverged, or produced a suspicious result.

Useful signals can be grouped as:

$$
O
=
(\text{metrics}, \text{logs}, \text{traces}, \text{artifacts}, \text{events})
$$

The goal is to connect symptoms to causes:

$$
\text{symptom}
\rightarrow
\text{evidence}
\rightarrow
\text{cause}
\rightarrow
\text{fix}
$$

## Key Ideas

- Training metrics explain learning behavior; system metrics explain resource behavior.
- Logs should identify stages and failures without leaking credentials or private paths.
- Metrics need units, sampling intervals, and labels that can be interpreted later.
- Data-validation failures and deployment events should be measured alongside service metrics.
- Artifacts such as configs, checkpoints, plots, and run summaries are part of observability.
- Alerts should point to actionable thresholds rather than create constant noise.

## Practical Checks

- What evidence proves the bottleneck: GPU, CPU, memory, storage, network, scheduler, or code?
- Are logs structured enough to compare runs?
- Are metrics aligned by timestamp with training steps or job events?
- Are model version, validation status, and rollout stage included in the evidence?
- Are failures classified rather than only recorded?
- Is public logging sanitized before it enters notes or dashboards?
- Can job state, logs, and artifacts be reconciled into one final outcome?
- Does each alert imply an owner and an action?

## Related

- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/data-validation|Data validation]]
- [[concepts/systems/deployment-strategy|Deployment strategy]]
- [[concepts/systems/training-run|Training run]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[infra/server-ops/incident-response|Incident response]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[logs/sanitization-checklist|Sanitization checklist]]
