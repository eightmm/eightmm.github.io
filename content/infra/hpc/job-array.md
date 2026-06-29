---
title: Job Array
tags:
  - infra
  - hpc
  - slurm
---

# Job Array

Job array는 한 번의 submission으로 비슷한 task를 많이 실행하는 방식입니다. Parameter sweep, dataset shard, docking batch, inference chunk, repeated evaluation job에 유용합니다.

Task를 $i$로 indexing하면:

$$
y_i = f(x_i; \theta)
\qquad
i \in \{1, \ldots, N\}
$$

각 array task는 같은 script template을 공유하면서 shard $x_i$ 하나 또는 configuration 하나를 처리합니다.

## Generic Slurm pattern

```bash
#SBATCH --array=1-<num-tasks>

TASK_ID="${SLURM_ARRAY_TASK_ID}"
```

Public note에서는 generic placeholder를 씁니다. Internal dataset path, private filename, cluster-specific account detail을 공개하지 않습니다.

## 쓸 때

- resource need가 비슷한 independent task가 많을 때.
- communication 없이 shard할 수 있는 workload일 때.
- 큰 job 하나를 다시 돌리는 것보다 failed shard만 retry하기 쉬울 때.
- 모든 task 완료 후 output을 merge할 수 있을 때.

## Risk

- 너무 많은 simultaneous task가 shared storage를 overload할 수 있습니다.
- per-task log가 inspect하기 어려워질 수 있습니다.
- shard imbalance가 resource를 낭비할 수 있습니다.
- accidental duplicate write가 output을 corrupt할 수 있습니다.

## 확인할 것

- 각 task가 independent한가?
- output path construction이 collision-free한가?
- concurrency limit이 storage와 scheduler policy에 적절한가?
- 모든 것을 recompute하지 않고 failed task ID만 rerun할 수 있는가?
- merge step이 deterministic하고 logged되는가?

## Related

- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/reproducibility|Reproducibility]]
