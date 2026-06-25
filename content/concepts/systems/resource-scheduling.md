---
title: Resource Scheduling
tags:
  - systems
  - infra
  - hpc
---

# Resource Scheduling

Resource scheduling is the process of assigning limited compute resources to jobs while balancing utilization, fairness, priority, and turnaround time.

A simple scheduling view is:

$$
j^\star
= \arg\max_{j \in \mathcal{Q}}
U(j, r_j, p_j, t_j)
$$

$\mathcal{Q}$ is the queue of waiting jobs, $r_j$ is the requested resource vector, $p_j$ is priority, $t_j$ is waiting time, and $U$ is the scheduler utility.

## Resource Vector

For a job $j$:

$$
r_j = (\mathrm{CPU}, \mathrm{GPU}, \mathrm{memory}, \mathrm{time}, \mathrm{storage}, \mathrm{network})
$$

The exact fields depend on the system, but the main idea is that the scheduler cannot place a job unless requested resources fit available capacity and policy constraints.

## Tradeoffs

- Utilization: keep expensive hardware busy.
- Fairness: avoid one user or project monopolizing shared capacity.
- Turnaround: reduce time from submission to completion.
- Fragmentation: avoid leaving unusable resource gaps.
- Reliability: handle failures, preemption, and retries.

## Checks

- Is the job over-requesting resources and blocking scheduling?
- Is the job under-requesting memory or time and failing mid-run?
- Is the workload better as one large job or many smaller jobs?
- Does the run need checkpointing before preemption or wall-time limits?
- Are scheduling policies described generically without private queue details?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/failure-recovery|Failure recovery]]
