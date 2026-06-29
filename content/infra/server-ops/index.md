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

- Use generic names such as `login-node`, `gpu-node`, and `shared-storage`.
- Replace exact paths with placeholders such as `/path/to/project`.
- Explain symptoms, diagnosis, and prevention without exposing private topology.
- Do not publish live security settings, user lists, credentials, ports, or hostnames.

## Runbook Shape

| Section | Write |
| --- | --- |
| Symptom | what the user or job observed |
| Scope | which generic layer is involved: account, storage, GPU, network, scheduler |
| Evidence | public-safe command class or log class, not private output |
| Immediate action | reversible or low-risk fix first |
| Root cause | what class of misconfiguration or resource failure caused it |
| Prevention | monitoring, backup, access boundary, documentation, smoke check |
| Sanitization | what must be removed before publishing |

## Notes

- [[infra/gpu/index#driver-and-cuda|GPU driver and CUDA debugging]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/account-management|Account and group management]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
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
