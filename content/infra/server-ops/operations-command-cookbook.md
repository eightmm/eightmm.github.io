---
title: Operations Command Cookbook
tags:
  - infra
  - server-ops
  - monitoring
---

# Operations Command Cookbook

This page collects public-safe command patterns for day-to-day server administration. It should explain what each command is for without publishing real hostnames, usernames, IP addresses, SSH ports, private paths, cluster names, or incident-specific output.

Use placeholders such as `<user>`, `<account>`, `<cluster>`, `<interface>`, `<node>`, `<job-id>`, and `<date>`.

$$
\text{command}
\rightarrow
\text{evidence class}
\rightarrow
\text{decision}
$$

## Triage Map

| Symptom | Start With | Then Route To |
| --- | --- | --- |
| Network feels slow | interface counters, socket pressure, packet drops | [[infra/hardware/storage-network|Storage and Network]] |
| Disk writes are slow | per-process IO and filesystem usage | [[infra/io/data-loading|Data loading and IO]] |
| Disk or RAID may be degraded | controller health and kernel logs | [[infra/server-ops/incident-response|Incident response]] |
| GPU disappears or crashes | driver logs, Xid class, device mapping | [[infra/gpu/index#driver-and-cuda|GPU driver and CUDA debugging]] |
| Slurm queue policy is unclear | association, QOS, TRES, fair-share | [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]] |
| Job resource request is wrong | `squeue`, `sacct`, script options | [[infra/hpc/slurm-job-script|Slurm Job Script]] |
| Access pattern looks unusual | aggregated auth-log summary | [[infra/server-ops/access-boundary|Access boundary]] |

## Monitoring Commands

| Need | Command pattern | Evidence |
| --- | --- | --- |
| Network interface traffic | `ip -s link show dev <interface>`, `sar -n DEV 1`, `ifstat`, `nload`, or a site-approved monitor | bandwidth pressure, packet drops, interface-level symptoms |
| Socket buffer pressure | `netstat -s`, `ss -s`, or platform-specific `netstat -m` | retransmit, buffer, or socket pressure symptoms |
| Process disk IO | `sudo iotop -o -a` | cumulative per-process read/write activity |
| GPU mapping | `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv` | index, UUID, PCI bus, and device identity |
| GPU Xid errors | `grep -i xid /var/log/kern.log` or `journalctl -k | grep -i xid` | driver-level GPU fault class |
| Storage controller health | `sudo <raid-tool> /c0 show` | disk or controller degradation class |

Do not treat a single command as a complete diagnosis. A useful incident note connects the command to a resource axis:

$$
\text{axis}
\in
\{\text{network}, \text{disk IO}, \text{storage health}, \text{GPU}, \text{scheduler}, \text{access}\}
$$

## Network Checks

For realtime network bandwidth, prefer interface counters or a local monitoring tool approved for the machine. `netstat -m` is platform-dependent and is not a portable Linux interface-bandwidth command.

```bash
ip -s link show dev <interface>
sar -n DEV 1
ifstat <interface>
```

Record public-safe observations as `interface saturation`, `packet drops`, or `network storage contention`. Do not publish real interface names if they reveal topology.

## Disk IO Checks

`iotop` is useful when a shared machine feels slow and you need to identify whether a process is doing heavy reads or writes.

```bash
sudo iotop -o -a
```

Use it to classify the issue, not to publish a process table. Public notes should replace usernames, commands, PIDs, project paths, and dataset names with generic categories.

| Raw output | Public wording |
| --- | --- |
| real process command | training job, data copy, archive extraction, checkpoint writer |
| real path | `/path/to/project` or storage class |
| real user | `<user>` or user group |
| exact timestamp | broad incident window or omit |

## Storage Health Checks

Controller tools such as `storcli` are appropriate for private runbooks. Public notes should describe the health class and use placeholders for controller IDs and tool paths.

```bash
sudo <raid-tool> /c<controller-id> show
```

If a command reports a degraded virtual drive, predicted failure, missing disk, or rebuild state, do not paste the raw output publicly. Summarize the status class and next safe action:

| Health class | Public action wording |
| --- | --- |
| degraded array | stop risky writes and escalate hardware maintenance |
| rebuild in progress | reduce IO-heavy workloads and monitor completion |
| media error | preserve evidence and plan disk replacement |
| controller warning | compare with kernel logs and vendor tool output |

## GPU Device Mapping

Prefer stable identifiers when mapping Slurm GRES, CUDA visible devices, and physical GPUs.

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
```

If minor numbers are required for a local runbook, collect them in a private note and publish only the generic mapping logic:

```bash
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  nvidia-smi -q -i "$i" | awk -v idx="$i" '
    /Product Name/ {sub(/.*: /,""); name=$0}
    /GPU UUID/     {sub(/.*: /,""); uuid=$0}
    /Bus Id/       {sub(/.*: /,""); bus=$0}
    /Minor Number/ {sub(/.*: /,""); minor=$0}
    END {printf("index=%s minor=%s dev=/dev/nvidia%s bus=%s uuid=%s name=%s\n", idx, minor, minor, bus, uuid, name)}
  '
done
```

Do not publish the real output if it reveals private node layout or asset inventory.

## GPU Xid Checks

GPU Xid errors are driver-level fault signals. They are useful for classifying hardware, driver, PCIe, power, thermal, or workload-triggered failures.

```bash
journalctl -k | grep -i xid
grep -i xid /var/log/kern.log
```

Public notes should keep only the error class and remediation pattern. Do not publish hostnames, bus IDs, GPU serial-like identifiers, usernames, exact incident times, or raw kernel-log excerpts.

## Slurm Read-Only Commands

Use these commands to inspect queue and accounting state without changing policy.

```bash
sacct -X --format="JobID,User,State%-10,JobName%-30,CPUTime,AllocTRES%-42"
squeue -o "%.18i %.10P %.10u %.20j %.8T %.10M %.6D %C %.10m %.6b %N %.10Q %.10p"
sprio
sshare -l
sacctmgr show association
```

For usage reporting, publish only sanitized trends unless the numbers are already public policy.

```bash
sreport job sizesbyaccount All_Clusters users=<user> account=<account> PrintJobCount start=<yyyy-mm-dd>
sreport -thourper cluster utilization --tres="cpu,mem,gres/gpu" start=<yyyy-mm-dd>
sreport cluster AccountUtilizationByUser cluster=<cluster> accounts=<account> start=<mm/dd/yy> -thourper
sreport user top -thourper start=<mm/dd/yy>
sacctmgr list transactions Action="Add Users" Start=<date-time> format=Where,Time
```

These commands belong conceptually in [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]]. Keep state-changing `sacctmgr` examples there with placeholders.

## Auth Log Summaries

For public notes, summarize the question and sanitize users/IPs. Do not publish real account names or source addresses.

Ubuntu versions can differ in auth log timestamp format. Keep version-specific commands private if they reveal operational patterns; publish an anonymized pattern:

```bash
grep "Accepted" /var/log/auth.log \
  | awk '{print "<minute>", "<user>", "(<source-ip>)"}' \
  | sort \
  | uniq -c
```

Public writeups should report the evidence class, not the raw access list:

| Raw evidence | Public wording |
| --- | --- |
| real username | `<user>` or user group |
| real source IP | `<source-ip>` or source category |
| real timestamp sequence | aggregated time window |
| real SSH port or endpoint | omit |

Private operations notes may keep version-specific parsing commands. Public notes should usually keep only the normalized pattern because log format differs across distributions.

## Network NAT Pattern

When documenting internet access for new nodes, avoid real interface names and topology. Use placeholders:

```bash
sudo iptables -t nat -A POSTROUTING -o <public-interface> -j MASQUERADE
```

Record the purpose and rollback path privately. Public notes should not expose real interface names, routing topology, firewall policy, or reachable subnets.

Because NAT and firewall changes are state-changing, public notes should frame them as an operations pattern:

$$
\text{need outbound access}
\rightarrow
\text{approved interface}
\rightarrow
\text{temporary rule}
\rightarrow
\text{rollback and documentation}
$$

## Safety Checks

- Is the command read-only, reversible, or state-changing?
- Does the command require `sudo`?
- Does raw output include users, IPs, hostnames, mount paths, serial numbers, or cluster topology?
- Can the output be summarized as an evidence class instead of pasted directly?
- Is this a one-off incident, or a reusable runbook pattern?

## Related

- [[infra/server-ops/index|Server Operations]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[infra/gpu/index|GPU]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/hpc/slurm|Slurm]]
- [[logs/sanitization-checklist|Sanitization checklist]]
