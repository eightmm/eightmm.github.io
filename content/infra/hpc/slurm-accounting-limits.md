---
title: Slurm Accounting and Limits
tags:
  - infra
  - hpc
  - slurm
---

# Slurm Accounting and Limits

Slurm accounting commands are useful for quota design, fair-share inspection, usage reporting, and user onboarding. Public notes must use placeholders and should never publish real account names, cluster names, usernames, QOS policy values, private partitions, or internal allocation rules.

$$
\text{association}
=
(\text{user}, \text{account}, \text{cluster}, \text{QOS}, \text{TRES limits})
$$

## Read-Only Inspection

| Need | Command pattern | Evidence |
| --- | --- | --- |
| Show submitted jobs with resources | `sacct -X --format="JobID,User,State%-10,JobName%-30,CPUTime,AllocTRES%-42"` | job state and allocated TRES |
| Show associations and limits | `sacctmgr show association` | user/account/QOS/TRES policy |
| Show fair-share detail | `sshare -l` | priority and fair-share inputs |
| Show submitted job priority | `sprio` | priority components |
| Show queue with priority field | `squeue -o "%.18i %.10P %.10u %.20j %.8T %.10M %.6D %C %.10m %.6b %N %.10Q %.10p"` | scheduler-visible job state |
| Show user creation transactions | `sacctmgr list transactions Action="Add Users" Start=<date-time> format=Where,Time` | administrative history |

Prefer read-only commands when writing public documentation. If output contains real users, accounts, partitions, nodes, or job names, summarize it instead of copying it.

## Usage Reports

Use reports to understand trend and allocation pressure, not to publish private lab usage.

```bash
sreport job sizesbyaccount All_Clusters users=<user> account=<account> PrintJobCount start=<yyyy-mm-dd>
sreport -thourper cluster utilization --tres="cpu,mem,gres/gpu" start=<yyyy-mm-dd>
sreport cluster AccountUtilizationByUser cluster=<cluster> accounts=<account> start=<mm/dd/yy> -thourper
sreport user top -thourper start=<mm/dd/yy>
```

Public writeups should convert exact results into safe summaries:

| Private output | Public summary |
| --- | --- |
| exact user ranking | workload concentration pattern |
| exact cluster/account name | generic account class |
| exact usage numbers | rounded or qualitative trend, only if safe |
| internal project name | omit or replace with public project category |

## Policy Changes

State-changing `sacctmgr` commands should stay in private operations notes unless converted into generic examples.

Use account classes, not real lab roles or usernames, in public documentation.

$$
\text{account class}
\rightarrow
\text{fair-share weight}
\rightarrow
\text{per-job limit}
\rightarrow
\text{group limit}
\rightarrow
\text{submit limit}
$$

## Example Account Classes

The values below are placeholders. They show the shape of a policy, not a recommended quota.

| Account class | Intended use | Typical constraints |
| --- | --- | --- |
| `<small-account>` | short onboarding, tutorial, or smoke-test workloads | low submit count, low concurrent jobs, modest group TRES |
| `<standard-account>` | regular research workloads | normal fair-share, higher submit count, bounded per-job TRES |
| `<long-account>` | long walltime or low-priority background runs | stricter submit count, explicit QOS |

```bash
sudo sacctmgr modify user where account=<account> \
  set MaxJobs=<n> MaxSubmit=<n> MaxTRES=cpu=<n>,gres/gpu=<n>,node=<n> GrpTRES=cpu=<n>,gres/gpu=<n>
```

```bash
sudo sacctmgr add user <user> account=<account> cluster=<cluster> \
  Fairshare=<n> GrpTRES="cpu=<n>,gres/gpu=<n>" \
  MaxJobs=<n> MaxTRES="cpu=<n>,gres/gpu=<n>,node=<n>" \
  MaxSubmitJobs=<n> DefaultAccount=<account> DefaultQOS=<qos> QOS=<qos>
```

```bash
sudo sacctmgr update qos set MaxSubmitJobsPerUser=<n> where Name=<qos>
```

## User Onboarding Pattern

Adding a user is not only an account database operation. It should connect identity, default account, QOS, and resource policy.

```bash
sudo sacctmgr add user <user> account=<account> cluster=<cluster> \
  Fairshare=<n> GrpTRES="cpu=<n>,gres/gpu=<n>" \
  MaxJobs=<n> MaxTRES="cpu=<n>,gres/gpu=<n>,node=<n>" \
  MaxSubmitJobs=<n> DefaultAccount=<account> DefaultQOS=<qos> QOS=<qos>
```

After the change, verify without publishing the raw association table:

```bash
sacctmgr show association user=<user> format=User,Account,Cluster,QOS,DefaultQOS,Fairshare,MaxJobs,MaxSubmitJobs,MaxTRES,GrpTRES
sshare -l
```

## Whole-Cluster Guardrails

Global user limits can protect the scheduler from accidental floods, but they are broad state-changing policy. Keep real values private.

```bash
sudo sacctmgr modify user set \
  MaxTRES=cpu=<n>,gres/gpu=<n>,node=<n> \
  GrpTRES=cpu=<n>,gres/gpu=<n> \
  MaxSubmit=<n> MaxJobs=<n>
```

Before applying a global limit, record:

| Check | Why |
| --- | --- |
| affected account classes | avoids surprising existing workloads |
| current queued/running jobs | prevents breaking active usage assumptions |
| QOS interaction | QOS limits may override or combine with association limits |
| rollback command | policy changes should be reversible |
| approval boundary | broad limits affect all users |

## Limit Design Questions

| Limit | Ask |
| --- | --- |
| `MaxJobs` | How many simultaneously running jobs should one user have? |
| `MaxSubmitJobs` / `MaxSubmit` | How many queued jobs can one user submit? |
| `MaxTRES` | What is the per-job ceiling for CPU, GPU, memory, or node count? |
| `GrpTRES` | What is the group-level aggregate ceiling? |
| `Fairshare` | How should long-term priority differ across account classes? |
| `QOS` | Which job duration, submit, or priority policy applies? |

## Common Mistakes

| Mistake | Result |
| --- | --- |
| Publishing real account names | reveals internal organization or project structure |
| Mixing `MaxSubmit`, `MaxSubmitJobs`, and QOS submit limits without checking precedence | users see confusing pending or submit failures |
| Setting per-job limits but forgetting group limits | one account can still consume too much aggregate resource |
| Setting group limits but forgetting per-job limits | one oversized job can dominate an allocation |
| Changing fair-share without checking historical usage | priority shifts can surprise active users |
| Copying raw `sacctmgr show association` output into public notes | leaks users, accounts, clusters, partitions, and policy |

## Public Boundary

- Use `<user>`, `<account>`, `<cluster>`, `<qos>`, `<partition>`, and `<date>`.
- Do not publish live quota values unless they are already official public policy.
- Do not publish real fair-share values, internal account names, user lists, or project allocations.
- Separate policy examples from actual operational state.
- Keep irreversible changes behind private review and administrative approval.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/server-ops/account-management|Account and group management]]
- [[infra/server-ops/operations-command-cookbook|Operations Command Cookbook]]
- [[logs/sanitization-checklist|Sanitization checklist]]
