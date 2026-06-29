---
title: Server Operations
tags:
  - infra
  - server-ops
---

# Server Operations

Server operation note는 Linux server, GPU driver, storage, account, networking, monitoring의 일반적인 troubleshooting pattern을 다룹니다.

공개 버전은 sanitize된 runbook처럼 읽혀야 합니다.

$$
\text{symptom}
\rightarrow
\text{evidence}
\rightarrow
\text{action}
\rightarrow
\text{prevention}
\rightarrow
\text{public boundary}
$$

## Public Writing Rules

- `login-node`, `gpu-node`, `shared-storage` 같은 generic name을 사용합니다.
- exact path는 `/path/to/project` 같은 placeholder로 바꿉니다.
- private topology를 노출하지 않고 symptom, diagnosis, prevention을 설명합니다.
- live security setting, user list, credential, port, hostname을 공개하지 않습니다.

## Runbook Shape

| Section | Write |
| --- | --- |
| Symptom | user 또는 job이 관찰한 것 |
| Scope | account, storage, GPU, network, scheduler 중 어떤 generic layer가 관련되는가 |
| Evidence | private output이 아니라 public-safe command class 또는 log class |
| Immediate action | reversible 또는 low-risk fix를 먼저 |
| Root cause | 어떤 misconfiguration 또는 resource failure class가 원인인가 |
| Prevention | monitoring, backup, access boundary, documentation, smoke check |
| Sanitization | publishing 전에 제거해야 하는 것 |

## Notes

- [[infra/server-ops/admin-usage-patterns|Admin Usage Patterns]]: daily server/HPC admin commands grouped by read-only checks, state-changing policy changes, and sensitive evidence boundaries
- [[infra/gpu/index#driver-and-cuda|GPU driver and CUDA debugging]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/account-management|Account and group management]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[infra/server-ops/operations-command-cookbook|Operations Command Cookbook]]: network, disk IO, GPU Xid, auth logs, Slurm inspection, and public-safe command patterns
- [[infra/server-ops/incident-response|Incident response]]
- [[infra/server-ops/backup-restore|Backup and restore]]
- [[infra/server-ops/access-boundary|Access boundary]]
- [[concepts/systems/environment-modules-containers|Environment modules and containers]]
- [[infra/gpu/index#utilization|GPU utilization]]

## Related

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/observability|Observability]]
- [[projects/index|Project index]]
- [[logs/index|Public logs]]
