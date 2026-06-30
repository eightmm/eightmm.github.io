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

## Public Network Boundary

Network and login notes are useful, but they are also where public pages most easily leak operational detail. Keep topology and exact endpoints private.

| Topic | Public-safe wording | Do not publish |
| --- | --- | --- |
| SSH access | login aggregation pattern or access class | host, IP, port, username, source IP |
| NAT/outbound access | placeholder interface and purpose | real interface, subnet, router policy |
| firewall change | state-changing boundary and rollback requirement | actual allowlist, exposed service, rule order |
| monitoring endpoint | signal class and owner | dashboard URL, port, hostname |
| auth logs | aggregated count after anonymization | raw `/var/log/auth.log` lines |

For example, publish the normalized aggregation idea, not a site-specific parser:

```bash
grep "Accepted" /var/log/auth.log \
  | awk '{print "<minute>", "<user>", "(<source-ip>)"}' \
  | sort \
  | uniq -c
```

For NAT examples, use placeholders only:

```bash
sudo iptables -t nat -A POSTROUTING -o <public-interface> -j MASQUERADE
```

The useful public claim is that outbound access required a controlled boundary change and a rollback path, not which machine or interface was used.

## Decision Table

| Resource | Main Risk | Public Check |
| --- | --- | --- |
| Dataset | Private data leaks into a broader group | State the intended audience and split read/write locations |
| Checkpoint | Large artifact is overwritten or copied without context | Record owner, run id, and lifecycle stage |
| Logs | Paths, usernames, prompts, or unpublished metrics leak | Sanitize before linking from public notes |
| Job queue | Users run under the wrong allocation or priority | Keep scheduler policy generic and link [Resource request](/infra/hpc/resource-request) |
| Published note | Internal operational detail becomes searchable | Review against [Sanitization checklist](/logs/sanitization-checklist) |

## Practical Checks

- Is access granted through groups instead of one-off permissions?
- Does write access imply responsibility for overwrite and deletion risk?
- Are dataset, checkpoint, and result directories separated by lifecycle?
- Does public documentation remove usernames, group names, private paths, hostnames, and ports?
- Is there an offboarding path that removes access without breaking reproducibility?
- Are public artifacts checked against [[logs/sanitization-checklist|Sanitization checklist]] before publishing?
- Are network examples reduced to placeholders rather than real endpoints?
- Is every auth-log example aggregated and anonymized?

## Failure Pattern

Most access incidents come from boundary drift: a temporary permission becomes normal, a copied artifact keeps old group ownership, or a public note inherits raw internal details. Treat access as part of the artifact lifecycle, not as a one-time command.

## Related

- [[infra/server-ops/account-management|Account and group management]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[infra/server-ops/incident-response|Incident response]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[projects/project-note-format|Project note format]]
