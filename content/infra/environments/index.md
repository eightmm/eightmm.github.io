---
title: Environments
unlisted: true
tags:
  - infra
  - environment
  - reproducibility
---

# Environments

Environment note는 module, container, dependency boundary, public-safe environment record를 다룹니다.

Environment는 run identity의 일부입니다.

$$
R = (\text{code}, \text{data}, \text{config}, \text{environment}, \text{seed})
$$

Compiler, CUDA stack, Python package set, container image, runtime mount가 바뀌면 model code가 그대로여도 결과가 달라질 수 있습니다.

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
