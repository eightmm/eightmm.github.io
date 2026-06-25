---
title: Server Operations
tags:
  - infra
  - server-ops
---

# Server Operations

Server operation notes cover general troubleshooting patterns for Linux servers, GPU drivers, storage, accounts, networking, and monitoring.

## Public Writing Rules

- Use generic names such as `login-node`, `gpu-node`, and `shared-storage`.
- Replace exact paths with placeholders such as `/path/to/project`.
- Explain symptoms, diagnosis, and prevention without exposing private topology.
- Do not publish live security settings, user lists, credentials, ports, or hostnames.

## Notes

- [[infra/server-ops/gpu-driver-cuda|GPU driver and CUDA debugging]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/account-management|Account and group management]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[infra/environment-modules-containers|Environment modules and containers]]
- [[infra/gpu-utilization|GPU utilization]]

## Related

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/observability|Observability]]
- [[projects/index|Project index]]
- [[logs/index|Public logs]]
