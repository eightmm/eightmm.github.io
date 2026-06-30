---
title: Monitoring Shared Machines
tags:
  - infra
  - server-ops
  - monitoring
---

# Monitoring Shared Machines

Monitoring tells you whether a shared research box is healthy and how its resources are being used before a job dies or a disk fills. The useful signals are GPU utilization and memory, host CPU and RAM, disk and quota, temperature, and process ownership.

An alert is useful only when it points to an action:

$$
\text{signal}
\rightarrow
\text{threshold}
\rightarrow
\text{owner}
\rightarrow
\text{action}
$$

For public notes, keep signals generic and remove live dashboard details.

## Signal Map

| Signal | Usually Indicates | First Question |
| --- | --- | --- |
| GPU memory high, utilization low | reserved but idle process, data stall, or blocked kernel | Is the process making progress? |
| CPU high, GPU low | preprocessing or dataloader bottleneck | Is input throughput limiting the step? |
| Disk near full | failed writes, broken checkpoints, or slow jobs | Which artifact class is growing? |
| IO wait high | shared storage contention or poor access pattern | Is data streamed from the wrong storage class? |
| Temperature or power throttling | hardware cooling or power limit | Is performance degraded across jobs? |

## Minimal Measurement Set

Shared-machine monitoring should start with a small cross-section of signals rather than a single dashboard number.

$$
\text{health snapshot}
=
\{\text{GPU}, \text{CPU}, \text{RAM}, \text{disk}, \text{network}, \text{scheduler}\}
$$

| Axis | Example signal | Public interpretation |
| --- | --- | --- |
| GPU | utilization, memory, Xid class | compute pressure, idle reservation, or driver fault |
| CPU/RAM | load, memory pressure, swap | preprocessing or host-side contention |
| Disk/IO | free space, IO wait, process IO | artifact growth or storage contention |
| Network | interface counters, packet drops | remote storage or distributed communication pressure |
| Scheduler | queue state, fair-share, priority | policy or resource-request bottleneck |

When documenting an incident publicly, replace live values with qualitative states such as `high`, `low`, `degraded`, `saturated`, or `unknown`.

## Command Families

The command belongs in [[infra/server-ops/operations-command-cookbook|Operations Command Cookbook]], but this page records what kind of monitoring question it answers.

| Question | Command family |
| --- | --- |
| Is network traffic or packet loss abnormal? | `ip -s`, `sar -n DEV`, `ifstat`, socket counters |
| Is one process dominating disk IO? | `iotop` or site-approved process IO monitor |
| Is storage health degraded? | controller health tool and kernel log class |
| Did a GPU driver fault occur? | `journalctl -k`, Xid grep, GPU mapping |
| Is Slurm priority or fair-share the issue? | `squeue`, `sprio`, `sshare`, `sacctmgr show association` |

Do not publish raw process tables, controller output, kernel logs, or scheduler tables. Summarize the signal and the next action.

## Practical Checks

- Watch GPU memory and utilization to spot idle reservations and runaway jobs.
- Alert on disk and quota thresholds before writes start failing.
- Track temperature and power; thermal throttling quietly slows training.
- Attribute heavy processes to a user or job so contention can be resolved.
- Keep dashboards internal — do not expose hostnames, ports, or live metrics publicly.
- Separate training metrics from system metrics so model behavior is not confused with resource contention.
- Save enough context to connect an incident note with [[concepts/systems/observability|Observability]] without exposing private systems.
- Pair system signals with scheduler state before blaming code or hardware.
- Turn repeated incidents into a sanitized runbook entry under [[infra/server-ops/admin-usage-patterns|Admin Usage Patterns]].

## Public Boundary

Use qualitative summaries such as `GPU memory pressure`, `storage contention`, or `quota risk`. Do not publish real dashboard URLs, hostnames, ports, process tables, usernames, live utilization, or incident timestamps that identify a private system.

## Related

- [[concepts/systems/observability|Observability]]
- [[infra/server-ops/incident-response|Incident response]]
- [[infra/gpu/index#driver-and-cuda|GPU driver and CUDA debugging]]
- [[infra/gpu/index|GPU]]
- [[infra/server-ops/index|Server operations]]
