---
title: Environment Management
tags:
  - systems
  - reproducibility
  - environment
---

# Environment Management

Environment management records and controls the software stack needed to reproduce a training run, inference job, or analysis. The environment includes language runtime, packages, CUDA stack, drivers, system libraries, and build flags.

A run depends on more than code:

$$
\text{run}
=
f(\text{code}, \text{data}, \text{config}, \text{environment}, \text{hardware})
$$

An environment version should answer:

$$
E
=
(\text{runtime}, \text{packages}, \text{accelerator stack}, \text{system libraries})
$$

## Key Ideas

- Package versions, CUDA versions, compiler behavior, and driver compatibility can change results or break runs.
- Lockfiles, containers, module snapshots, and setup scripts serve different levels of reproducibility.
- Environment capture should happen before a long run starts, not after a result looks interesting.
- Public notes should describe reproducibility patterns, not private machine details.

## Environment Contract

An environment record should be enough to explain compatibility:

$$
E
=
(\text{OS class},
\text{runtime},
\text{package lock},
\text{accelerator runtime},
\text{driver/API compatibility},
\text{build flags})
$$

This should be paired with the code commit, data version, and config. Environment metadata alone does not reproduce a result.

## Capture Points

Capture the environment at three times:

$$
E_{\mathrm{start}},
\quad
E_{\mathrm{checkpoint}},
\quad
E_{\mathrm{inference}}
$$

The environment used for final inference or evaluation can differ from the training launch environment, especially after checkpoint resume, container rebuilds, or dependency upgrades.

## Failure Modes

- Optional acceleration library is missing, silently changing speed or numerics.
- Package resolver installs different transitive versions later.
- Training and inference environments use different preprocessing libraries.
- A checkpoint cannot be loaded because model code or dependency versions drifted.
- Public notes expose private paths or machine identifiers instead of reproducible environment classes.

## Practical Checks

- Can the environment be reconstructed on another machine?
- Are Python, CUDA, PyTorch, compiler, and driver compatibility recorded?
- Are optional acceleration libraries enabled or silently missing?
- Does the run record include environment metadata with code and data versions?
- Are secrets, private paths, and internal hostnames excluded from public logs?
- Is the environment captured before the run starts?
- Are training, evaluation, and inference environments the same or intentionally different?
- Are lockfiles, container images, or module snapshots tied to the run artifact?

## Related

- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/model-versioning|Model versioning]]
- [[concepts/data/data-versioning|Data versioning]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/gpu/index#driver-and-cuda|GPU driver and CUDA debugging]]
