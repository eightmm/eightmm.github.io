---
title: Systems
tags:
  - ai
  - systems
---

# Systems

AI Systems는 모델이 실제 run, artifact, service, workflow가 되는 지점을 다룹니다. 모델 구조나 loss 자체가 아니라, 학습이 어떻게 기록되고, 추론이 어떤 계약으로 실행되며, 결과가 어떻게 재현 가능한 주장으로 남는지를 봅니다.

이 페이지는 Infra가 아닙니다. Infra는 GPU, Slurm, storage, server operation처럼 실행 환경의 제약을 다루고, Systems는 모델 run과 artifact가 어떤 계약으로 관리되는지를 다룹니다.

$$
\text{AI system}
=
(\text{model}, \text{data}, \text{runtime}, \text{artifact}, \text{operation})
$$

## Main Areas

| Area | Use for | Canonical notes |
| --- | --- | --- |
| Training | run state, checkpoint, scaling, failure recovery | [Training run](/concepts/systems/training-run), [Checkpoint state](/concepts/systems/checkpoint-state), [Distributed training](/concepts/systems/distributed-training) |
| Inference | prediction-time execution, batch/online mode, output contract | [Inference](/concepts/systems/inference), [Batch and online inference](/concepts/systems/batch-online-inference), [Inference contract](/concepts/systems/inference-contract) |
| Serving | endpoint, batching, capacity, latency, rollout | [Model serving](/concepts/systems/model-serving), [Inference serving](/concepts/systems/inference-serving), [Inference capacity planning](/concepts/systems/inference-capacity-planning), [Latency and throughput](/concepts/systems/latency-throughput) |
| Environment | module, container, dependency, runtime drift | [Environment management](/concepts/systems/environment-management), [Environment modules and containers](/concepts/systems/environment-modules-containers) |
| Reproducibility | run record, artifact, versioning, evidence boundary | [Reproducibility](/concepts/systems/reproducibility), [Run artifact](/concepts/systems/run-artifact), [Model versioning](/concepts/systems/model-versioning) |
| Operations | monitoring, deployment, recovery, resource scheduling | [Deployment strategy](/concepts/systems/deployment-strategy), [Observability](/concepts/systems/observability), [Failure recovery](/concepts/systems/failure-recovery), [Resource scheduling](/concepts/systems/resource-scheduling) |

## Boundary With Infra

| Question | Route |
| --- | --- |
| training 또는 inference concept인가? | [AI Systems](/ai/systems), [Systems concepts](/concepts/systems) |
| hardware, GPU, Slurm, storage, server operation이 관련되는가? | [Infra](/infra) |
| run 또는 artifact가 reproducible하다는 public record가 필요한가? | [Reproducibility](/concepts/systems/reproducibility), [Run record](/infra/reproducibility/run-record) |
| cluster 또는 shared machine에서 실패했는가? | [HPC](/infra/hpc), [GPU](/infra/gpu), [Server operations](/infra/server-ops) |

## Reading Path

1. [[concepts/systems/training-run|Training run]]으로 run state와 checkpoint 경계를 봅니다.
2. [[concepts/systems/inference|Inference]]와 [[concepts/systems/model-serving|Model serving]]으로 실행/serving 경계를 봅니다.
3. [[concepts/systems/environment-management|Environment management]]로 dependency와 runtime drift를 정리합니다.
4. [[concepts/systems/reproducibility|Reproducibility]]와 [[concepts/systems/run-artifact|Run artifact]]로 공개 가능한 증거 형식을 맞춥니다.
5. 실제 GPU/HPC/storage 문제가 되면 [[infra/index|Infra]]로 내려갑니다.

## System Artifact Checklist

AI system note는 모델 이름보다 artifact boundary를 먼저 남깁니다.

| Artifact | Include |
| --- | --- |
| Dataset snapshot | source, version, split, preprocessing contract |
| Model checkpoint | architecture, weight, tokenizer/featurizer, config |
| Training state | optimizer, scheduler, step, seed, environment |
| Inference contract | input schema, output schema, batching, limit |
| Evaluation record | metric, selection rule, baseline, uncertainty |
| Runtime environment | package version, container/module, hardware class |

## Related

- [[ai/index|AI]]
- [[concepts/systems/index|AI systems concepts]]
- [[infra/index|Infra]]
- [[projects/index|Projects]]
