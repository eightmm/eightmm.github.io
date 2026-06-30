---
title: Sanitization Checklist
tags:
  - logs
  - privacy
  - publishing
---

# Sanitization Checklist

Sanitization converts private working notes into public notes. The goal is to keep the transferable lesson while removing details that identify systems, people, unpublished work, or security posture.

## Remove

- Server IPs, hostnames, node names, account names, and SSH connection details.
- Private paths, mount topology, credentials, tokens, keys, and secrets.
- Internal project names, collaborator details, and unpublished experiment results.
- Exact firewall, router, or security configuration.
- Private dataset names, sample identifiers, and non-public benchmarks.

## Replace With

- Generic names such as `login-node`, `gpu-node`, `shared-storage`, and `/path/to/project`.
- Resource classes instead of private partition or queue names.
- Public dataset names only when the dataset is actually public.
- General failure modes instead of private run logs.

## Checks

- Could this note help an outsider without revealing private infrastructure?
- Would a collaborator be comfortable seeing the context public?
- Does the note avoid security-sensitive configuration?
- Are uncertain claims marked as unresolved?
- Has the note passed [[inbox/publishing-gate|Publishing gate]] before promotion?
- Has the note's destination been checked with [[logs/log-promotion-rule|Log promotion rule]]?

## Related

- [[logs/public-log-format|Public log format]]
- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/log-promotion-rule|Log promotion rule]]
- [[inbox/publishing-gate|Publishing gate]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[infra/index|Infra]]
- [[posts/essays/blog-and-wiki-workflow|Blog and wiki workflow]]
