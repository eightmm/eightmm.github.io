---
title: HPC Research Workflows
tags:
  - projects
  - infra
  - hpc
---

# HPC Research Workflows

This project collects public, generalized patterns for running research workloads on shared GPU and HPC systems. It focuses on reproducibility, debugging, and safe operations rather than private cluster details.

## Artifact

The artifact is a reusable workflow for designing, submitting, monitoring, and closing research compute runs.

## Problem

Research runs often fail because the workflow around the model is weak: unclear resource requests, missing checkpoints, untracked environments, and poor failure logs.

## Public Boundary

Keep the note generic. Do not include real server names, IPs, account names, SSH ports, private mount paths, queue names that reveal infrastructure, user lists, or private job IDs.

## Workflow

1. Run a local or small-batch smoke test.
2. Submit a constrained job with explicit CPU, GPU, memory, and time assumptions.
3. Save checkpoints and logs in a reproducible layout.
4. Record code commit, environment, seed, and dataset version.
5. Debug failures from scheduler state, application logs, and hardware symptoms.

## Checks

- Can the run be reproduced from public information without private infrastructure details?
- Are resource assumptions explicit enough to explain failures?
- Are checkpoints frequent enough for the expected wall time?
- Is the result a public method note, a private experiment, or a post candidate?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[infra/distributed-training|Distributed training]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/project-note-format|Project note format]]
