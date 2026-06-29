---
title: Environment Modules and Containers
unlisted: true
tags:
  - infra
  - environment
---

# Environment Modules and Containers

Environment modules and containers are systems concepts when they define reproducibility, and infra concepts when they touch cluster policy, GPU runtime, storage mounts, or server operations.

The public note should answer one operational question:

$$
\text{same code}
+ \text{same environment}
+ \text{same input}
\rightarrow
\text{reproducible run}
$$

If the environment can silently change between login shell, scheduler job, notebook, and container runtime, the result is not reproducible even when the code is unchanged.

| Question | Go To |
| --- | --- |
| What are modules and containers? | [Environment modules and containers](/concepts/systems/environment-modules-containers) |
| How should an environment be captured? | [Environment management](/concepts/systems/environment-management) |
| What belongs in a run record? | [Reproducible run record](/infra/reproducibility/run-record) |
| Is this a scheduler job script issue? | [Slurm job script](/infra/hpc/slurm-job-script) |

## Infra Checklist

| Check | Why it matters |
| --- | --- |
| Module list | compiler, CUDA, MPI, Python, and library ABI can change behavior |
| Container image | image tag, digest, CUDA base, and entrypoint define the runtime |
| Host binding | GPU driver, filesystem mount, UID/GID, and network access come from the host |
| Scheduler shell | interactive environment and batch environment may load different defaults |
| Artifact capture | run records should include image/module state, not only code commit |

## Boundary

| Belongs here | Belongs elsewhere |
| --- | --- |
| module/container policy, image provenance, mount behavior, GPU runtime compatibility | model objective, architecture, dataset semantics |
| scheduler job environment, runtime failure, dependency drift | paper claim, benchmark metric, learning method |

Public examples should use placeholders only. Do not publish private registry names, internal mount paths, hostnames, usernames, project paths, tokens, or cluster-specific module trees.
