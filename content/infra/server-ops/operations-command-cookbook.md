---
title: Operations Command Cookbook
tags:
  - infra
  - server-ops
  - monitoring
---

# Operations Command Cookbook

This page collects public-safe command patterns for day-to-day server administration. It should explain what each command is for without publishing real hostnames, usernames, IP addresses, SSH ports, private paths, cluster names, or incident-specific output.

Use placeholders such as `<user>`, `<account>`, `<cluster>`, `<interface>`, `<job-id>`, and `<date>`.

$$
\text{command}
\rightarrow
\text{evidence class}
\rightarrow
\text{decision}
$$

## Monitoring Commands

| Need | Command pattern | Evidence |
| --- | --- | --- |
| Network interface traffic | `netstat -m` or a site-approved interface monitor | bandwidth pressure, socket buffer pressure, network symptoms |
| Process disk IO | `sudo iotop -o -a` | cumulative per-process read/write activity |
| GPU mapping | `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv` | index, UUID, PCI bus, and device identity |
| GPU Xid errors | `grep -i xid /var/log/kern.log` or `journalctl -k | grep -i xid` | driver-level GPU fault class |
| Storage controller health | `sudo <raid-tool> /c0 show` | disk or controller degradation class |

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

## Network NAT Pattern

When documenting internet access for new nodes, avoid real interface names and topology. Use placeholders:

```bash
sudo iptables -t nat -A POSTROUTING -o <public-interface> -j MASQUERADE
```

Record the purpose and rollback path privately. Public notes should not expose real interface names, routing topology, firewall policy, or reachable subnets.

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
