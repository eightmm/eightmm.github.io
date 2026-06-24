---
title: Account and Group Management
tags:
  - infra
  - server-ops
  - accounts
---

# Account and Group Management

On a shared machine, access control is mostly about users and groups: who can read a dataset, write to a shared directory, or submit jobs under a project allocation. Clear group structure prevents both accidental data loss and over-broad access.

## Practical Checks

- Grant access through groups, not per-user permissions that drift over time.
- Apply least privilege — most users do not need administrative rights.
- Remove access promptly when someone leaves a project.
- Keep dataset and checkpoint directories group-owned so collaborators can read them.
- Never publish real usernames, group names, or credentials in public notes.

## Related

- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/index|Server operations]]
- [[infra/index|Infra]]
