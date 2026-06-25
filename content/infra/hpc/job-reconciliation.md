---
title: Job Reconciliation
tags:
  - infra
  - hpc
  - reliability
---

# Job Reconciliation

Job reconciliation is the step after submission where the recorded run state is compared with scheduler state, logs, checkpoints, and output artifacts. It prevents duplicate relaunches and stale "probably running" assumptions.

A reconciled job state can be represented as:

$$
R_j =
(\text{scheduler state}, \text{exit code}, \text{logs}, \text{artifacts}, \text{run record})
$$

The job is not operationally closed until these pieces agree.

## Why It Matters

- Long jobs often finish after the person or agent that launched them has stopped watching.
- A scheduler terminal state does not prove the output artifact is complete.
- A missing log line does not prove failure if the job wrote a valid checkpoint.
- Relaunching without reconciliation can duplicate expensive work.

## Generic Reconciliation Flow

1. Query scheduler state for the job or job array.
2. Classify the terminal state: completed, failed, cancelled, timeout, out-of-memory, preempted, or unknown.
3. Inspect the final log lines and resource summary.
4. Check output artifacts and completion markers.
5. Record the outcome in the run record.
6. Decide whether to resume, retry, rerun with different resources, or stop.

## Outcome Table

| Scheduler State | Artifact State | Interpretation |
| --- | --- | --- |
| Completed | Complete | Close the run and record verification. |
| Completed | Missing or partial | Treat as artifact failure, not success. |
| Failed or timeout | Valid checkpoint | Resume if the protocol allows it. |
| Failed or timeout | No checkpoint | Diagnose before relaunch. |
| Unknown | Unknown | Do not report completion. Gather more evidence. |

## Public Boundary

Public notes should describe the failure class and recovery lesson without publishing private job IDs, queue names, hostnames, account names, internal paths, user names, or unpublished results.

## Checks

- Is the scheduler state terminal?
- Does the output have a completion marker?
- Is the checkpoint compatible with the current code and config?
- Did the run record capture the final state and recovery decision?
- Is the public summary sanitized?

## Related

- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/observability|Observability]]
