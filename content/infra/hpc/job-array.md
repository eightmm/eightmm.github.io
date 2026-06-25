---
title: Job Array
tags:
  - infra
  - hpc
  - slurm
---

# Job Array

A job array runs many similar tasks from one submission. It is useful for parameter sweeps, dataset shards, docking batches, inference chunks, or repeated evaluation jobs.

If tasks are indexed by $i$:

$$
y_i = f(x_i; \theta)
\qquad
i \in \{1, \ldots, N\}
$$

Each array task processes one shard $x_i$ or one configuration, while sharing the same script template.

## Generic Slurm Pattern

```bash
#SBATCH --array=1-<num-tasks>

TASK_ID="${SLURM_ARRAY_TASK_ID}"
```

Use generic placeholders in public notes. Do not publish internal dataset paths, private filenames, or cluster-specific account details.

## When To Use

- Many independent tasks with similar resource needs.
- Workloads that can be sharded without communication.
- Retrying failed shards is easier than retrying one large job.
- Output can be merged after all tasks complete.

## Risks

- Too many simultaneous tasks can overload shared storage.
- Per-task logs can become hard to inspect.
- Shard imbalance can waste resources.
- Accidental duplicate writes can corrupt outputs.

## Checks

- Is each task independent?
- Is output path construction collision-free?
- Is the concurrency limit appropriate for storage and scheduler policy?
- Can failed task IDs be rerun without recomputing everything?
- Is the merge step deterministic and logged?

## Related

- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/reproducibility|Reproducibility]]
