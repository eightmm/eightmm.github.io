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

## Account Class Contract

Shared HPC 운영에서는 Unix 계정, group permission, Slurm association이 서로 다른 layer입니다. Public note에서는 실제 이름을 숨기고 아래 contract만 남깁니다.

$$
\text{person}
\rightarrow
\text{login identity}
\rightarrow
\text{Unix groups}
\rightarrow
\text{scheduler association}
\rightarrow
\text{artifact ownership}
$$

| Layer | Decides | Public placeholder |
| --- | --- | --- |
| Login identity | machine access and audit trail | `<user>` |
| Unix group | filesystem read/write boundary | `<group>` |
| Slurm account | allocation, fair-share, submit policy | `<account>` |
| QOS | walltime, priority, submission limits | `<qos>` |
| Artifact ownership | who can maintain outputs after a run | workload class |

Do not collapse these layers in notes. A user can have filesystem access but wrong scheduler limits, or a valid Slurm association but no permission to read the dataset.

## Onboarding Checklist

Before adding or changing a user, define the intended access class rather than copying another user's settings.

| Question | Why |
| --- | --- |
| Which data or project artifacts should be readable? | prevents over-broad group access |
| Which directories should be writable? | prevents accidental overwrite of shared outputs |
| Which scheduler account/QOS should be default? | prevents jobs landing in the wrong policy class |
| What is the first smoke job or read-only verification? | catches broken permission before real work |
| What should happen at offboarding? | avoids orphaned artifacts and stale access |

Public examples should use `<user>`, `<group>`, `<account>`, and `<qos>`. Exact names, policy values, and approval history stay private.

## Verification Pattern

After an account or group change, verify through read-only commands and summarize the evidence class.

```bash
id <user>
groups <user>
sacctmgr show association user=<user> format=User,Account,Cluster,QOS,DefaultQOS
sshare -l
```

The public note should say `group membership and scheduler association were checked`, not paste the real output.

## Practical Checks

- Grant access through groups, not per-user permissions that drift over time.
- Apply least privilege — most users do not need administrative rights.
- Remove access promptly when someone leaves a project.
- Keep dataset and checkpoint directories group-owned so collaborators can read them.
- Never publish real usernames, group names, or credentials in public notes.
- Separate account lifecycle from artifact lifecycle: a run record should remain interpretable after a user leaves.
- Review access when a project changes from private work to public artifact.
- Keep Slurm policy changes in [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]] and filesystem permission notes here.

## What Not to Publish

Public notes can describe account lifecycle and group-ownership patterns. They should not include real usernames, group names, UID/GID values, home paths, hostnames, SSH details, admin membership, or live policy exceptions.

## Related

- [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/access-boundary|Access boundary]]
- [[infra/server-ops/admin-usage-patterns|Admin Usage Patterns]]
- [[infra/server-ops/index|Server operations]]
- [[infra/index|Infra]]
