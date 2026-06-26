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

The scheduler sees this as a placement problem:

$$
\operatorname{place}(j)
\quad\text{only if}\quad
r_j \le R_{\mathrm{available}}
$$

Oversized requests can wait longer even if the job itself is simple. Undersized requests can fail, OOM, or run slowly.

## Generic Slurm Fields

```bash
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --gres=gpu:<gpu-count>
#SBATCH --mem=<memory>
#SBATCH --time=<hh:mm:ss>
```

These are placeholders. Do not publish private partitions, accounts, hostnames, internal paths, or cluster-specific names.

## Sizing From a Smoke Run

Measure a small run before requesting a full run:

$$
\tau_{\mathrm{full}}
\approx
\tau_{\mathrm{smoke}}
\times
\frac{N_{\mathrm{full}}}{N_{\mathrm{smoke}}}
\times
\alpha
$$

where $\alpha>1$ is a safety margin for IO, checkpointing, startup, validation, and variance in throughput.

Memory can be estimated similarly:

$$
m_{\mathrm{request}}
=
\max_t m_{\mathrm{observed}}(t)
\times
\beta
$$

where $\beta$ is a safety margin. The margin should be justified by observed variability, not used to hide unknown behavior.

## Bottleneck Matching

Resource requests should match the bottleneck:

- CPU-bound: preprocessing, compression, data transforms, some feature extraction.
- GPU-bound: dense tensor training, inference, docking kernels, geometric models.
- Memory-bound: large batches, large graphs, high-resolution structures, data joins.
- IO-bound: many small files, remote storage, checkpoint storms, dataset sharding.
- Scheduler-bound: too many tiny jobs or oversized monolithic jobs.

Requesting more GPUs does not help if data loading or CPU preprocessing is the bottleneck.

## Array vs Monolith

For independent shards, prefer job arrays:

$$
W
=
\bigcup_{k=1}^{K} W_k,
\qquad
W_i \cap W_j = \varnothing
$$

This makes failures smaller, improves scheduling flexibility, and reduces the cost of rerunning one failed shard.

## Practical Heuristics

- Start with a small smoke run before a full training or screening job.
- Increase memory only when logs or monitoring show memory pressure.
- Request wall time based on measured iteration speed, not a guess.
- Separate CPU-heavy preprocessing from GPU-heavy training when possible.
- Record the final public-safe resource shape in [[infra/reproducibility/run-record|Reproducible run record]].
- Keep GPU count, batch size, data-loader workers, and checkpoint interval consistent with measured throughput.
- For public writeups, describe resource class generically instead of naming private cluster resources.

## Checks

- Does the request match the bottleneck: CPU, GPU, memory, IO, or time?
- Is the job too large to schedule quickly?
- Can the workload be split into [[infra/hpc/job-array|job arrays]]?
- Does the job checkpoint before wall-time limits?
- Are cluster-specific values removed from public notes?
- Was the request derived from a smoke run or previous measured run?
- Is the wall-time estimate compatible with validation, checkpointing, and cleanup?
- Is the request reproducible from the run record without exposing private infrastructure?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-array|Job array]]
- [[infra/gpu/utilization|GPU utilization]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/gpu/memory|GPU memory]]
