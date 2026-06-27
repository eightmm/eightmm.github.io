---
title: Environments
unlisted: true
tags:
  - infra
  - environment
  - reproducibility
---

# Environments

Environment notes cover modules, containers, dependency boundaries, and public-safe environment records.

The environment is part of the run identity:

$$
R = (\text{code}, \text{data}, \text{config}, \text{environment}, \text{seed})
$$

Changing the compiler, CUDA stack, Python package set, container image, or runtime mount can change the result even when the model code is unchanged.

## Scope

- Software stacks used for training, inference, preprocessing, and analysis.
- Environment modules, containers, lockfiles, image digests, and runtime wrappers.
- Driver/runtime boundaries such as host driver versus container user space.
- Public-safe environment records that explain reproducibility without exposing private machines.

## Notes

- [[concepts/systems/environment-modules-containers|Environment modules and containers]]

## Checks

- Can the environment be reconstructed from public-safe metadata?
- Are CUDA, compiler, Python, package, and container versions recorded at the right granularity?
- Is the host-specific part separated from portable user-space dependencies?
- Are private registry names, mount paths, credentials, and hostnames excluded?

## Where New Notes Go

- Module-versus-container decisions go here.
- Dependency pinning and environment drift go here.
- Cluster-specific execution details go under [[infra/hpc/index|HPC]].
- Incident-style environment failures go under [[infra/server-ops/index|Server operations]] if the note is operational.

## Related

- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducibility/run-record|Reproducible run record]]
