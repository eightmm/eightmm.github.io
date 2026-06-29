---
title: AI Systems
tags:
  - systems
  - machine-learning
---

# AI Systems

AI systems note는 model이 어떻게 학습되고, serving되고, 측정되고, 재현되는지 설명합니다. 위치상 model concept과 infrastructure operation 사이에 있습니다.

핵심 systems 질문은 다음과 같습니다.

$$
\text{quality} = f(\text{data}, \text{model}, \text{compute}, \text{workflow}, \text{evaluation})
$$

Model은 함수 $f_\theta$만이 아닙니다. 동시에 training process, inference process, operational artifact입니다.

## 주제

| Area | Use for | Start |
| --- | --- | --- |
| Training process | training run, checkpoint state, distributed run behavior | [Training run](/concepts/systems/training-run), [Checkpoint state](/concepts/systems/checkpoint-state), [Distributed training](/concepts/systems/distributed-training) |
| Inference and serving | batch/online inference, serving path, latency, throughput, capacity | [Inference](/concepts/systems/inference), [Model serving](/concepts/systems/model-serving), [Inference serving](/concepts/systems/inference-serving) |
| Contracts and artifacts | input/output contract, model card, versioning, run artifact | [Inference contract](/concepts/systems/inference-contract), [Model card](/concepts/systems/model-card), [Run artifact](/concepts/systems/run-artifact) |
| Reproducibility | environment, modules, containers, experiment tracking, public run records | [Environment management](/concepts/systems/environment-management), [Environment modules and containers](/concepts/systems/environment-modules-containers), [Reproducibility](/concepts/systems/reproducibility) |
| Scaling and resources | scheduling, memory/compute tradeoff, storage IO, scaling claim | [Resource scheduling](/concepts/systems/resource-scheduling), [Memory-compute tradeoff](/concepts/systems/memory-compute-tradeoff), [Scaling claim contract](/concepts/systems/scaling-claim-contract) |
| Deployment and reliability | deployment strategy, observability, failure recovery, data validation | [Deployment strategy](/concepts/systems/deployment-strategy), [Observability](/concepts/systems/observability), [Failure recovery](/concepts/systems/failure-recovery) |
| Infra bridge | hardware speed, storage/network, scheduler reconciliation | [Memory hierarchy](/infra/hardware/memory-hierarchy), [Storage and network](/infra/hardware/storage-network), [Job reconciliation](/infra/hpc/job-reconciliation) |

## Systems vs Infra

| 질문 | Systems에서 볼 것 | Infra에서 볼 것 |
| --- | --- | --- |
| 모델을 어떻게 실행 단위로 만들까? | [Training run](/concepts/systems/training-run), [Inference](/concepts/systems/inference) | [HPC](/infra/hpc), [GPU](/infra/gpu) |
| serving capacity를 어떻게 잡을까? | [Inference capacity planning](/concepts/systems/inference-capacity-planning), [Latency and throughput](/concepts/systems/latency-throughput) | [GPU memory](/infra/gpu), [Hardware](/infra/hardware) |
| 결과를 나중에 검증할 수 있을까? | [Run artifact](/concepts/systems/run-artifact), [Experiment lifecycle](/concepts/systems/experiment-lifecycle) | [Reproducibility](/infra/reproducibility) |
| environment 문제가 재현성에 영향을 주나? | [Environment management](/concepts/systems/environment-management) | [Server operations](/infra/server-ops), [HPC](/infra/hpc) |
| bottleneck이 어디인가? | [Memory-compute tradeoff](/concepts/systems/memory-compute-tradeoff), [Storage and IO](/concepts/systems/storage-io) | [GPU](/infra/gpu), [Storage and IO](/infra/io) |

## 확인할 것

| Check | Why |
| --- | --- |
| bottleneck이 data loading, compute, memory, communication, scheduling 중 무엇인가? | fix 위치를 code, data, hardware, scheduler 중 하나로 좁힙니다 |
| workload가 single-device, data-parallel, sharded, pipeline, model-parallel 중 무엇인가? | scaling claim과 resource request가 달라집니다 |
| 목표가 model quality, time-to-train, cost, latency, throughput, reliability 중 무엇인가? | 같은 system도 최적화 기준이 다르면 설계가 바뀝니다 |
| run artifact가 commit, config, seed, dataset version, environment를 포함하는가? | 나중에 metric이나 failure를 다시 확인할 수 있습니다 |
| training metric, validation metric, serving metric이 분리되어 있는가? | offline 성능과 runtime 품질을 혼동하지 않습니다 |
| deployment path가 evaluation과 같은 preprocessing과 constraint를 보존하는가? | 평가와 실제 실행 사이의 hidden mismatch를 줄입니다 |
| public-facing artifact에 model card와 inference contract가 있는가? | 사용 범위, input/output, failure format을 명확히 합니다 |

## Related

- [[ai/index|AI]]
- [[ai/systems|Systems]]
- [[infra/index|Infra]]
- [[papers/systems/index|Systems papers]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/research-methodology/index|Research methodology]]
- [[agents/verification/verification-loop|Verification loop]]
