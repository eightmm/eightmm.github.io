---
title: Storage Mounts and Permissions
tags:
  - infra
  - server-ops
  - storage
---

# Storage Mounts and Permissions

Shared research machines mix local disks, network filesystems, and scratch space, each with different speed, persistence, and permission rules. Many "file not found" or "permission denied" failures are really an unmounted share or a group-ownership gap, not a code bug.

## Practical Checks

- Confirm the expected path is actually mounted before debugging the program.
- Know which storage is fast-but-volatile (scratch) and which is durable (project/home).
- Use group ownership and the setgid bit so shared directories stay writable by the team.
- Watch quota and free space — silent write failures often trace to a full volume.
- Use placeholders such as `/path/to/project` in public notes; never publish real topology.

## Related

- [[infra/server-ops/index|Server operations]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/index|Infra]]
