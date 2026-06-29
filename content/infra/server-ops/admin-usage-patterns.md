---
title: Admin Usage Patterns
tags:
  - infra
  - server-ops
  - hpc
---

# Admin Usage Patterns

이 페이지는 shared GPU/HPC server를 운영하면서 자주 쓰는 점검 명령을 공개 가능한 형태로 정리하는 입구입니다. 실제 운영 노트에는 더 구체적인 값이 있을 수 있지만, public page에는 host, user, account, cluster, IP, SSH port, private path, incident output을 남기지 않습니다.

운영 명령은 먼저 세 종류로 나눕니다.

| Class | Meaning | Public rule |
| --- | --- | --- |
| read-only check | 상태 확인, queue 확인, log 검색 | command pattern은 공개 가능, output은 sanitize |
| state-changing admin | user/QOS/limit/NAT 변경 | placeholder만 사용하고 실제 정책값은 비공개 |
| sensitive evidence | auth log, kernel log, RAID output, process table | raw output을 붙이지 않고 evidence class로 요약 |

$$
\text{command}
\rightarrow
\text{evidence}
\rightarrow
\text{decision}
\rightarrow
\text{sanitized note}
$$

## Quick Routing

| Need | Use | Detail page |
| --- | --- | --- |
| 네트워크 대역폭 또는 socket pressure 확인 | `ip`, `sar`, `ifstat`, `ss`, `netstat` | [[infra/hardware/storage-network|Storage and Network]] |
| 프로세스별 disk IO 확인 | `sudo iotop -o -a` | [[infra/hardware/storage-network#storage-health-evidence|Storage Health Evidence]] |
| RAID/disk health 확인 | `sudo <raid-tool> /c<controller-id> show` | [[infra/server-ops/incident-response|Incident response]] |
| GPU Xid 또는 device mapping 확인 | `journalctl`, `grep`, `nvidia-smi` | [[infra/gpu/index#xid-triage|Xid Triage]] |
| Slurm queue, usage, fair-share 확인 | `sacct`, `squeue`, `sreport`, `sshare`, `sprio` | [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]] |
| Slurm user/account/QOS limit 변경 | `sacctmgr add/modify/update` | [[infra/hpc/slurm-accounting-limits#policy-changes|Policy Changes]] |
| `sbatch` script option 확인 | `#SBATCH ...` | [[infra/hpc/slurm-job-script|Slurm Job Script]] |
| 접속 패턴 집계 | sanitized `auth.log` aggregation | [[infra/server-ops/access-boundary|Access boundary]] |
| NAT 또는 outbound access 설정 | `iptables` pattern | [[infra/server-ops/access-boundary|Access boundary]] |

## Host and IO Checks

Network interface counters are useful when storage, distributed training, or remote service calls feel slow. Do not publish real interface names if they reveal topology.

```bash
ip -s link show dev <interface>
sar -n DEV 1
ifstat <interface>
ss -s
netstat -s
```

`netstat -m` is platform-specific. On Linux, prefer interface counters for bandwidth and `ss -s` or `netstat -s` for socket-level symptoms.

For per-process disk IO:

```bash
sudo iotop -o -a
```

Public summary should say `checkpoint writer`, `archive extraction`, `dataset reader`, or `large copy`, not the real command, path, user, or project.

For disk/controller health:

```bash
sudo <raid-tool> /c<controller-id> show
```

Raw controller output may include device layout, serial-like identifiers, and incident details. Public notes should only keep the health class: `healthy`, `degraded`, `rebuild`, `media error`, or `controller warning`.

## GPU Checks

Use stable identifiers when debugging GPU allocation, scheduler mapping, or CUDA-visible device confusion.

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
```

When local device minor numbers matter, keep the raw result private and publish only the pattern:

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

For Xid triage:

```bash
journalctl -k | grep -i xid
grep -i xid /var/log/kern.log
```

Public notes should report the class of failure, not raw kernel lines.

| Evidence class | Ask next |
| --- | --- |
| repeated Xid on one GPU | same physical device, same workload, or same thermal/power window? |
| Xid after driver change | driver/runtime compatibility or reboot/module state? |
| Xid across many GPUs | host, driver, PCIe, power, or scheduler-wide event? |
| mapping mismatch | UUID, PCI bus, minor number, Slurm GRES, and `CUDA_VISIBLE_DEVICES` alignment |

## Access Log Aggregation

Access logs are sensitive. Public documentation should keep the aggregation idea, not real users or source IPs.

```bash
grep "Accepted" /var/log/auth.log \
  | awk '{print "<minute>", "<user>", "(<source-ip>)"}' \
  | sort \
  | uniq -c
```

Distribution log formats differ. If a private runbook needs version-specific parsing, keep it private and convert public notes to this normalized form:

| Raw field | Public replacement |
| --- | --- |
| username | `<user>` or account class |
| source IP | `<source-ip>` or source category |
| exact login minute | aggregated window |
| endpoint/port | omit |

## Slurm Read-Only Inspection

Use these before changing policy.

```bash
sacct -X --format="JobID,User,State%-10,JobName%-30,CPUTime,AllocTRES%-42"
squeue -o "%.18i %.10P %.10u %.20j %.8T %.10M %.6D %C %.10m %.6b %N %.10Q %.10p"
sprio
sshare -l
sacctmgr show association
sacctmgr list transactions Action="Add Users" Start=<date-time> format=Where,Time
```

Usage reports:

```bash
sreport job sizesbyaccount All_Clusters users=<user> account=<account> PrintJobCount start=<yyyy-mm-dd>
sreport -thourper cluster utilization --tres="cpu,mem,gres/gpu" start=<yyyy-mm-dd>
sreport cluster AccountUtilizationByUser cluster=<cluster> accounts=<account> start=<mm/dd/yy> -thourper
sreport user top -thourper start=<mm/dd/yy>
```

Public writeups should describe trend and bottleneck, not user rankings, account names, or exact internal allocation.

## Slurm Policy Changes

These examples show command shape only. They are not recommended quotas.

Add a user to a generic account class:

```bash
sudo sacctmgr add user <user> account=<account> cluster=<cluster> \
  Fairshare=<n> GrpTRES="cpu=<n>,gres/gpu=<n>" \
  MaxJobs=<n> MaxTRES="cpu=<n>,gres/gpu=<n>,node=<n>" \
  MaxSubmitJobs=<n> DefaultAccount=<account> DefaultQOS=<qos> QOS=<qos>
```

Modify account-class limits:

```bash
sudo sacctmgr modify user where account=<account> \
  set MaxJobs=<n> MaxSubmit=<n> \
  MaxTRES=cpu=<n>,gres/gpu=<n>,node=<n> \
  GrpTRES=cpu=<n>,gres/gpu=<n>
```

Modify a broad user limit:

```bash
sudo sacctmgr modify user set \
  MaxTRES=cpu=<n>,gres/gpu=<n>,node=<n> \
  GrpTRES=cpu=<n>,gres/gpu=<n> \
  MaxSubmit=<n> MaxJobs=<n>
```

Limit QOS submissions per user:

```bash
sudo sacctmgr update qos set MaxSubmitJobsPerUser=<n> where Name=<qos>
```

Before applying state-changing commands, record the intended policy, affected account class, rollback plan, and verification command in a private admin note.

## Network NAT Pattern

NAT/firewall changes are state-changing and topology-sensitive. Public notes should use placeholders only.

```bash
sudo iptables -t nat -A POSTROUTING -o <public-interface> -j MASQUERADE
```

Keep real interface names, subnets, router assumptions, and security policy private.

## `sbatch` Option Reference

Common options belong in [[infra/hpc/slurm-job-script|Slurm Job Script]], but the operational split is:

| Group | Options | Risk |
| --- | --- | --- |
| identity | `--job-name`, `--output`, `--error` | leaking project names or private paths |
| resources | `--time`, `--nodes`, `--cpus-per-task`, `--mem`, `--gres` | over-requesting or queue delay |
| placement | `--partition`, `--nodelist`, `--nodefile`, `--exclusive` | exposing cluster topology |
| environment | `--export` | leaking tokens or private settings |
| workflow | `--dependency`, `--begin`, `--mail-type`, `--mail-user` | hidden ordering assumptions or personal info |

## Public Note Checklist

- Are all users, IPs, ports, hostnames, cluster names, account names, internal paths, and private project names removed?
- Is this command read-only or state-changing?
- If state-changing, is it shown only with placeholders?
- Is raw output replaced with evidence class and decision?
- Does the note link to the right durable concept page instead of becoming a private incident dump?

## Related

- [[infra/server-ops/operations-command-cookbook|Operations Command Cookbook]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]]
- [[infra/hpc/slurm-job-script|Slurm Job Script]]
- [[infra/gpu/index|GPU]]
- [[infra/hardware/storage-network|Storage and network]]
- [[logs/sanitization-checklist|Sanitization checklist]]
