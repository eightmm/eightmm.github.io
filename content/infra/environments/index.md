---
title: Environments
unlisted: true
tags:
  - infra
  - environment
  - reproducibility
---

# Environments

Environment notes connect software stacks, dependency boundaries, and public-safe run records. Reusable definitions live under [[concepts/systems/index|AI Systems]]; infra pages focus on scheduler, GPU runtime, storage mount, or server-operation boundaries.

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

## Routing

| Question | Go To |
| --- | --- |
| What are modules and containers? | [Environment modules and containers](/concepts/systems/environment-modules-containers) |
| How should a run environment be captured? | [Environment management](/concepts/systems/environment-management), [Reproducibility](/concepts/systems/reproducibility) |
| Is the issue a scheduler job script or cluster launch? | [HPC](/infra/hpc), [Slurm job script](/infra/hpc/slurm-job-script) |
| Is the issue an operational failure? | [Server operations](/infra/server-ops), [Incident response](/infra/server-ops/incident-response) |

## Related

- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducibility/run-record|Reproducible run record]]
