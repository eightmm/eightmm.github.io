---
title: Storage Mounts and Permissions
tags:
  - infra
  - server-ops
  - storage
---

# Storage Mounts and Permissions

Shared research machines mix local disks, network filesystems, and scratch space, each with different speed, persistence, and permission rules. Many "file not found" or "permission denied" failures are really an unmounted share or a group-ownership gap, not a code bug.

A storage choice can be summarized by:

$$
\text{storage class}
=
(\text{latency}, \text{throughput}, \text{capacity}, \text{persistence}, \text{sharing})
$$

Training code should not treat these classes as interchangeable.

## Storage Classes

| Class | Good For | Risk |
| --- | --- | --- |
| Local scratch | temporary features, cache, short-lived intermediates | data loss after cleanup or node failure |
| Shared project storage | durable configs, manifests, selected outputs | contention and permission drift |
| Home directory | small personal config and notes | quota pressure and poor sharing semantics |
| Object/archive storage | large durable artifacts or releases | slower random access and restore delay |

## Practical Checks

- Confirm the expected path is actually mounted before debugging the program.
- Know which storage is fast-but-volatile (scratch) and which is durable (project/home).
- Use group ownership and the setgid bit so shared directories stay writable by the team.
- Watch quota and free space — silent write failures often trace to a full volume.
- Use placeholders such as `/path/to/project` in public notes; never publish real topology.
- Record whether an artifact is durable, scratch, recomputable, or backed up.
- Test restore for irreplaceable metadata instead of assuming a copy is usable.

## Debugging Order

1. Confirm the path exists and the expected storage class is mounted.
2. Check read/write permission at the directory where the job writes outputs.
3. Check quota, free space, and inode availability before changing code.
4. Run a small read/write smoke check with a placeholder path.
5. Record the artifact class in [[infra/reproducibility/run-record|Reproducible run record]].

## Related

- [[infra/server-ops/index|Server operations]]
- [[infra/server-ops/backup-restore|Backup and restore]]
- [[infra/server-ops/access-boundary|Access boundary]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/index|Infra]]
