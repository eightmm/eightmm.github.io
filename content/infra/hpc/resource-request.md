---
title: Resource Request
tags:
  - infra
  - hpc
  - slurm
---

# Resource Request

A resource request describes what a job needs from a shared cluster: CPUs, GPUs, memory, wall time, storage, and sometimes special constraints. It is the contract between the job and the scheduler.

For a job $j$:

$$
r_j = (c_j, g_j, m_j, \tau_j)
$$

$c_j$ is CPU count, $g_j$ is GPU count, $m_j$ is memory, and $\tau_j$ is requested wall time.

## Generic Slurm Fields

```bash
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --gres=gpu:<gpu-count>
#SBATCH --mem=<memory>
#SBATCH --time=<hh:mm:ss>
```

These are placeholders. Do not publish private partitions, accounts, hostnames, internal paths, or cluster-specific names.

## Practical Heuristics

- Start with a small smoke run before a full training or screening job.
- Increase memory only when logs or monitoring show memory pressure.
- Request wall time based on measured iteration speed, not a guess.
- Separate CPU-heavy preprocessing from GPU-heavy training when possible.
- Record the final public-safe resource shape in [[infra/reproducible-run-record|Reproducible run record]].

## Checks

- Does the request match the bottleneck: CPU, GPU, memory, IO, or time?
- Is the job too large to schedule quickly?
- Can the workload be split into [[infra/hpc/job-array|job arrays]]?
- Does the job checkpoint before wall-time limits?
- Are cluster-specific values removed from public notes?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/gpu-memory|GPU memory]]
