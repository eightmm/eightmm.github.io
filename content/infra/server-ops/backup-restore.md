---
title: Backup and Restore
tags:
  - infra
  - server-ops
  - storage
  - reliability
---

# Backup and Restore

Backup is only useful when restore has been tested. In research infrastructure, the important question is not "is there a copy?" but "can the right artifact be recovered before the decision becomes stale?"

Two common targets are:

$$
\text{RPO} = t_{\text{failure}} - t_{\text{last recoverable copy}}
$$

$$
\text{RTO} = t_{\text{service restored}} - t_{\text{failure}}
$$

where RPO is the maximum acceptable data loss window and RTO is the maximum acceptable recovery time.

## What to Back Up

- Small durable artifacts: configs, scripts, manifests, metadata, and notes.
- Recomputable large artifacts: checkpoints, embeddings, intermediate features, and predictions.
- Irreplaceable data: raw public inputs, labels, curated splits, and annotation metadata.
- Operational state: environment manifests, scheduler scripts, service configs, and run records.

## Restore Drill

A restore drill should answer:

1. Can the artifact be located by version, date, run id, or manifest?
2. Can checksums or file counts detect a partial restore?
3. Can a small downstream command read the restored artifact?
4. Can permissions be restored without granting broader access?
5. Is the restore procedure documented without private paths or credentials?

## Public Boundary

Public notes can describe retention classes, validation checks, and recovery reasoning. They should not publish storage topology, backup hostnames, exact schedules, encryption keys, cloud bucket names, private paths, or live restore commands for internal systems.

## Related

- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/incident-response|Incident response]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[infra/reproducible-run-record|Reproducible run record]]
