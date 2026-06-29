---
title: Account and Group Management
tags:
  - infra
  - server-ops
  - accounts
---

# Account and Group Management

On a shared machine, access control is mostly about users and groups: who can read a dataset, write to a shared directory, or submit jobs under a project allocation. Clear group structure prevents both accidental data loss and over-broad access.

The public abstraction is:

$$
\text{identity}
\rightarrow
\text{group}
\rightarrow
\text{permission}
\rightarrow
\text{resource}
$$

Do not document the real identities or group names. Document the pattern.

## Lifecycle

| Stage | Check |
| --- | --- |
| Onboarding | Add the user to the minimal project group needed for the work |
| Active work | Keep shared outputs group-owned and separate from personal scratch |
| Artifact handoff | Ensure run records, configs, and manifests remain readable after ownership changes |
| Offboarding | Remove access without deleting interpretable public-safe records |
| Publication | Replace real usernames, group names, and internal paths with placeholders |

## Practical Checks

- Grant access through groups, not per-user permissions that drift over time.
- Apply least privilege — most users do not need administrative rights.
- Remove access promptly when someone leaves a project.
- Keep dataset and checkpoint directories group-owned so collaborators can read them.
- Never publish real usernames, group names, or credentials in public notes.
- Separate account lifecycle from artifact lifecycle: a run record should remain interpretable after a user leaves.
- Review access when a project changes from private work to public artifact.

## What Not to Publish

Public notes can describe account lifecycle and group-ownership patterns. They should not include real usernames, group names, UID/GID values, home paths, hostnames, SSH details, admin membership, or live policy exceptions.

## Related

- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/access-boundary|Access boundary]]
- [[infra/server-ops/index|Server operations]]
- [[infra/index|Infra]]
