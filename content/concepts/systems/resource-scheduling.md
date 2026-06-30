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

## Request vs Allocation

Users request resources, but the scheduler allocates concrete resources.

$$
r_{\mathrm{request}}
\rightarrow
r_{\mathrm{allocated}}
\rightarrow
r_{\mathrm{used}}
$$

| Term | Meaning | Failure mode |
| --- | --- | --- |
| requested | what the job asks for | over-request delays queue; under-request fails run |
| allocated | what scheduler grants | mismatch with expected device/layout |
| used | what workload actually consumes | low utilization or OOM |

Good run records compare all three. A job can be scheduled successfully and still be inefficient or invalid for the experiment.

## Scheduling Signals

| Signal | Interpretation |
| --- | --- |
| pending reason | policy, dependency, resources, priority, reservation |
| wait time | contention or oversized request |
| utilization | whether allocated resources are doing useful work |
| preemption | run needs checkpoint and resume contract |
| exit state | scheduler-level completion, not application correctness |
| fair-share/priority | long-term allocation policy input |

Scheduler state is evidence about execution, not proof that the model result is correct.

## Tradeoffs

- Utilization: keep expensive hardware busy.
- Fairness: avoid one user or project monopolizing shared capacity.
- Turnaround: reduce time from submission to completion.
- Fragmentation: avoid leaving unusable resource gaps.
- Reliability: handle failures, preemption, and retries.

## Workload Shape

| Workload | Scheduling pattern |
| --- | --- |
| many independent items | job array or batch queue |
| long training run | checkpointing and wall-time-aware resume |
| distributed training | co-scheduled nodes/GPUs and communication-aware placement |
| interactive debugging | short allocation or dev partition |
| batch inference | shard by input and retry failed shards |

Choosing job shape is part of system design. One huge job may be slower to schedule and harder to recover than many bounded shards.

## Checks

- Is the job over-requesting resources and blocking scheduling?
- Is the job under-requesting memory or time and failing mid-run?
- Is the workload better as one large job or many smaller jobs?
- Does the run need checkpointing before preemption or wall-time limits?
- Are scheduling policies described generically without private queue details?
- Are requested, allocated, and actually used resources recorded separately?
- Is scheduler success separated from application-level success?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/job-array|Job array]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/failure-recovery|Failure recovery]]
