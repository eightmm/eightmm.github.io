---
title: Systems
tags:
  - ai
  - systems
---

# Systems

AI Systems는 모델이 실제 run, artifact, service, workflow가 되는 지점을 다룹니다. 모델 구조나 loss 자체가 아니라, 학습이 어떻게 기록되고, 추론이 어떤 계약으로 실행되며, 결과가 어떻게 재현 가능한 주장으로 남는지를 봅니다.

$$
\text{AI system}
=
(\text{model}, \text{data}, \text{runtime}, \text{artifact}, \text{operation})
$$

## Main Areas

| Area | Use For | Canonical Notes |
| --- | --- | --- |
| Training | run state, checkpoint, scaling, failure recovery | [Training run](/concepts/systems/training-run), [Checkpoint state](/concepts/systems/checkpoint-state), [Distributed training](/concepts/systems/distributed-training) |
| Inference | prediction-time execution, batch/online modes, output contracts | [Inference](/concepts/systems/inference), [Batch and online inference](/concepts/systems/batch-online-inference), [Inference contract](/concepts/systems/inference-contract) |
| Serving | endpoints, batching, capacity, latency, rollout | [Model serving](/concepts/systems/model-serving), [Inference serving](/concepts/systems/inference-serving), [Inference capacity planning](/concepts/systems/inference-capacity-planning), [Latency and throughput](/concepts/systems/latency-throughput) |
| Environment | modules, containers, dependencies, runtime drift | [Environment management](/concepts/systems/environment-management), [Environment modules and containers](/concepts/systems/environment-modules-containers) |
| Reproducibility | run records, artifacts, versioning, evidence boundary | [Reproducibility](/concepts/systems/reproducibility), [Run artifact](/concepts/systems/run-artifact), [Model versioning](/concepts/systems/model-versioning) |
| Operations | monitoring, deployment, recovery, resource scheduling | [Deployment strategy](/concepts/systems/deployment-strategy), [Observability](/concepts/systems/observability), [Failure recovery](/concepts/systems/failure-recovery), [Resource scheduling](/concepts/systems/resource-scheduling) |

## Boundary With Infra

| Question | Route |
| --- | --- |
| What is the training or inference concept? | [AI Systems](/ai/systems), [Systems concepts](/concepts/systems) |
| What hardware, GPU, Slurm, storage, or server operation is involved? | [Infra](/infra) |
| What public record proves a run or artifact is reproducible? | [Reproducibility](/concepts/systems/reproducibility), [Run record](/infra/reproducibility/run-record) |
| What failed on a cluster or shared machine? | [HPC](/infra/hpc), [GPU](/infra/gpu), [Server operations](/infra/server-ops) |

## Reading Path

1. [[concepts/systems/training-run|Training run]]으로 run state와 checkpoint 경계를 봅니다.
2. [[concepts/systems/inference|Inference]]와 [[concepts/systems/model-serving|Model serving]]으로 실행/serving 경계를 봅니다.
3. [[concepts/systems/environment-management|Environment management]]로 dependency와 runtime drift를 정리합니다.
4. [[concepts/systems/reproducibility|Reproducibility]]와 [[concepts/systems/run-artifact|Run artifact]]로 공개 가능한 증거 형식을 맞춥니다.
5. 실제 GPU/HPC/storage 문제가 되면 [[infra/index|Infra]]로 내려갑니다.

## Related

- [[ai/index|AI]]
- [[concepts/systems/index|AI systems concepts]]
- [[infra/index|Infra]]
- [[projects/index|Projects]]
