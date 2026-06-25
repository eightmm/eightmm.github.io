---
title: Access Boundary
tags:
  - infra
  - server-ops
  - security
---

# Access Boundary

An access boundary defines who can read, write, execute, submit, administer, or publish a resource. On shared research machines, the boundary should follow the project and artifact lifecycle rather than individual convenience.

For a resource $r$ and action $a$, access can be written as:

$$
\operatorname{allow}(u, a, r)
\iff
\exists g \in G(u): \operatorname{policy}(g, a, r)=1
$$

where $u$ is a user, $G(u)$ is the set of groups for that user, and `policy` records whether a group is allowed to perform action $a$ on resource $r$.

## Boundary Types

- Read boundary: who can inspect data, logs, checkpoints, or notes.
- Write boundary: who can modify shared artifacts or overwrite outputs.
- Execute boundary: who can run jobs, services, or scripts.
- Admin boundary: who can change users, groups, quotas, drivers, or services.
- Publish boundary: what can leave the private environment and enter public notes.

## Practical Checks

- Is access granted through groups instead of one-off permissions?
- Does write access imply responsibility for overwrite and deletion risk?
- Are dataset, checkpoint, and result directories separated by lifecycle?
- Does public documentation remove usernames, group names, private paths, hostnames, and ports?
- Is there an offboarding path that removes access without breaking reproducibility?
- Are public artifacts checked against [[logs/sanitization-checklist|Sanitization checklist]] before publishing?

## Related

- [[infra/server-ops/account-management|Account and group management]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/incident-response|Incident response]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[projects/project-note-format|Project note format]]
