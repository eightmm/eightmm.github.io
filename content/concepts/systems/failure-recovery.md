---
title: Failure Recovery
tags:
  - systems
  - reliability
  - reproducibility
---

# Failure Recovery

Failure recovery is the ability to continue, retry, or diagnose a workflow after interruption. In AI systems, failures include preemption, out-of-memory errors, corrupt data, worker crashes, network issues, invalid model outputs, and dependency drift.

A recoverable workflow records enough state to define:

$$
\operatorname{Recover}(F_t, S_t)
\rightarrow
S_{t'}
$$

where $F_t$ is a failure event and $S_t$ is the last trusted workflow state.

## Recovery Levels

- Retry one item: useful for batch inference or data processing.
- Resume one run: useful for training jobs with checkpoints.
- Restart one service: useful for serving with health checks and model reload.
- Reproduce one failure: useful for debugging and public postmortems.
- Roll back one version: useful when a model or preprocessing update is bad.

## Failure Record

A useful failure record includes:

- What failed.
- When it failed.
- Last completed step or item.
- Error class and short relevant message.
- Code/config/data/model version.
- Resource context such as memory, batch size, or input size.
- Recovery action and whether it worked.

Do not include private paths, hostnames, account names, credentials, or unpublished results in public failure records.

## Checks

- Is the workflow idempotent under retry?
- Can partial outputs be detected and skipped or rewritten safely?
- Is there a last-known-good checkpoint or model version?
- Does the retry policy avoid hiding systematic errors?
- Can failures be summarized publicly without leaking internal infrastructure?
- Is terminal job state reconciled before retrying or claiming recovery?
- Is the recovery plan tied to a backup, checkpoint, or last-known-good artifact?

## Related

- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/observability|Observability]]
- [[infra/server-ops/incident-response|Incident response]]
- [[infra/server-ops/backup-restore|Backup and restore]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
