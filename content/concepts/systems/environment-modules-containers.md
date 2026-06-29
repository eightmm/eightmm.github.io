---
title: Environment Modules and Containers
aliases:
  - infra/environment-modules-containers
tags:
  - systems
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

The key boundary is host versus user space:

$$
\text{host}
=
(\text{kernel}, \text{driver}, \text{scheduler}, \text{mounts})
$$

$$
\text{user space}
=
(\text{compiler}, \text{runtime}, \text{packages}, \text{application})
$$

Containers package much of user space, but they still depend on host drivers, filesystem mounts, and scheduler integration.

## Modules

Modules are useful when the site provides CUDA, compilers, MPI, and libraries centrally. They are lightweight but depend on cluster configuration.

Checks:

- Record loaded module names and versions.
- Avoid assuming the same default module set across machines.
- Confirm that Python packages and binary libraries resolve to the intended stack.

## Module Contract

| Field | Why It Matters |
| --- | --- |
| module names and versions | identifies compiler, CUDA, MPI, and shared libraries |
| load order | later modules can override paths |
| shell context | login shell and batch shell may load different defaults |
| environment variables | `PATH`, library paths, and runtime flags change behavior |
| Python environment | package resolver and binary wheels may not match module ABI |

For public notes, record the class of stack rather than the private module tree. A useful public phrase is `CUDA-compatible module stack with recorded compiler/runtime versions`, not a site-specific module path.

## Containers

Containers improve portability by packaging user-space dependencies. They still depend on host drivers, runtime support, storage mounts, and scheduler integration.

Checks:

- Record container image name, digest, and build recipe.
- Keep secrets and private data out of the image.
- Verify GPU access, mounts, permissions, and working directory behavior.

## Container Contract

| Field | Why It Matters |
| --- | --- |
| image tag and digest | tag alone can move; digest pins content |
| base image | controls OS user-space and CUDA runtime family |
| build recipe | explains how the image was produced |
| entrypoint and command | affects shell behavior and working directory |
| mounted paths | controls data, logs, checkpoints, and permissions |
| GPU runtime | host driver must support the container runtime |
| UID/GID behavior | affects write permission and shared artifacts |

Container reproducibility is strongest when image digest, code commit, config, and data version are recorded together.

## Driver and Runtime Boundary

GPU containers do not carry the kernel driver. A typical stack is:

$$
\text{host driver}
\rightarrow
\text{container CUDA runtime}
\rightarrow
\text{framework build}
\rightarrow
\text{application kernels}
$$

If a run fails after moving between machines, check this chain before changing model code.

| Symptom | Possible Boundary |
| --- | --- |
| GPU not visible | scheduler allocation, container runtime, device binding |
| CUDA initialization fails | driver/runtime compatibility |
| import works but kernels fail | framework build and compiled extension mismatch |
| run works interactively but not in job | batch shell, module load, mount, or environment variable |
| checkpoint cannot load | package, model code, or serialization version drift |

## Choosing Modules or Containers

| Need | Prefer | Caveat |
| --- | --- | --- |
| site-provided MPI or scheduler integration | modules | less portable across clusters |
| reproducible Python package stack | container or lockfile | still depends on host driver and mounts |
| quick interactive debugging | modules or lightweight env | defaults may differ from batch jobs |
| long public reproduction | container digest plus run record | avoid private registry or data paths |
| custom compiled extensions | container or explicit build script | ABI must match CUDA/compiler stack |

## Run Record Fields

A public-safe run record should include enough environment detail to explain compatibility without exposing infrastructure.

| Field | Public-Safe Form |
| --- | --- |
| OS class | `Linux x86_64` or similar class |
| accelerator stack | CUDA/runtime family, framework build |
| package environment | lockfile, frozen package list, or image digest |
| module/container state | generic module list or image digest |
| hardware class | GPU class or CPU class, not private hostname |
| launch mode | interactive, batch job, containerized, distributed |
| known drift | training/evaluation/inference environment differences |

## Public Notes

- Describe generic environment strategy.
- Do not publish private registry names, internal mount paths, credentials, or host-specific module trees.
- Do not paste full environment dumps if they include usernames, private paths, tokens, internal registries, hostnames, or mount topology.
- Prefer summarized compatibility fields and keep full private logs outside the public blog.

## Checks

- Is this a systems concept, or an operational failure that belongs in [[infra/server-ops/index|Server operations]]?
- Can another reader understand the runtime boundary without private cluster details?
- Are training, evaluation, and inference using the same environment or intentionally different ones?
- Is the host driver/runtime boundary recorded for GPU work?
- Are module defaults and batch-job environment differences accounted for?
- Is the container pinned by digest rather than only by tag when reproducibility matters?

## Related

- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/gpu/index#driver-and-cuda|GPU driver and CUDA debugging]]
