---
title: Environment Modules and Containers
aliases:
  - infra/environment-modules-containers
tags:
  - infra
  - environment
  - reproducibility
---

# Environment Modules and Containers

Environment modules and containers are two ways to control software environments on research machines and HPC clusters. Modules expose shared system software; containers package a larger user-space environment.

The environment should be treated as part of the run contract:

$$
(\text{code}, \text{data}, \text{config}, \text{environment})
\rightarrow
\text{result}
$$

## Modules

Modules are useful when the site provides CUDA, compilers, MPI, and libraries centrally. They are lightweight but depend on cluster configuration.

Checks:

- Record loaded module names and versions.
- Avoid assuming the same default module set across machines.
- Confirm that Python packages and binary libraries resolve to the intended stack.

## Containers

Containers improve portability by packaging user-space dependencies. They still depend on host drivers, runtime support, storage mounts, and scheduler integration.

Checks:

- Record container image name, digest, and build recipe.
- Keep secrets and private data out of the image.
- Verify GPU access, mounts, permissions, and working directory behavior.

## Public Notes

- Describe generic environment strategy.
- Do not publish private registry names, internal mount paths, credentials, or host-specific module trees.

## Related

- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/server-ops/gpu-driver-cuda|GPU driver and CUDA debugging]]
