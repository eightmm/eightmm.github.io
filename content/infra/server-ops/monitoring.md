---
title: Monitoring Shared Machines
tags:
  - infra
  - server-ops
  - monitoring
---

# Monitoring Shared Machines

Monitoring tells you whether a shared research box is healthy and how its resources are being used before a job dies or a disk fills. The useful signals are GPU utilization and memory, host CPU and RAM, disk and quota, temperature, and process ownership.

## Practical Checks

- Watch GPU memory and utilization to spot idle reservations and runaway jobs.
- Alert on disk and quota thresholds before writes start failing.
- Track temperature and power; thermal throttling quietly slows training.
- Attribute heavy processes to a user or job so contention can be resolved.
- Keep dashboards internal — do not expose hostnames, ports, or live metrics publicly.

## Related

- [[infra/server-ops/gpu-driver-cuda|GPU driver and CUDA debugging]]
- [[infra/gpu|GPU]]
- [[infra/server-ops/index|Server operations]]
