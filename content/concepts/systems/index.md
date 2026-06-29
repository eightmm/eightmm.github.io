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

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/inference-serving|Inference serving]]
- [[concepts/systems/inference-capacity-planning|Inference capacity planning]]
- [[concepts/systems/deployment-strategy|Deployment strategy]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/distributed-training-runbook|Distributed training runbook]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/environment-modules-containers|Environment modules and containers]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/model-versioning|Model versioning]]
- [[concepts/systems/data-validation|Data validation]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[infra/hardware/storage-network|Storage and network]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]

## 확인할 것

- bottleneck이 data loading, compute, memory, communication, scheduling 중 무엇인가?
- workload가 single-device, data-parallel, sharded, pipeline, model-parallel training 중 무엇을 요구하는가?
- scheduler 관점에서 resource request, job size, queue time, preemption risk가 맞게 잡혔는가?
- environment와 storage path가 run record의 일부인가?
- 목표가 model quality, time-to-train, cost, latency, throughput, reliability 중 무엇인가?
- 논문이 scaling을 주장한다면 quality, data, model size, compute budget, runtime boundary가 분리되어 있는가?
- experiment lifecycle이 question에서 claim까지 기록되는가?
- run artifact가 나중에 inspection이나 metric check를 하기에 충분한가?
- commit, config, seed, dataset version, environment로 run을 재현할 수 있는가?
- workflow가 preemption, partial output, service failure에서 회복 가능한가?
- terminal run을 log, scheduler state, artifact로 reconcile할 수 있는가?
- training metric, validation metric, serving metric이 분리되어 있는가?
- deployment path가 evaluation과 같은 preprocessing과 constraint를 보존하는가?
- public-facing artifact에 model card와 inference contract가 있는가?
- output을 하나의 model version, validation boundary, rollout decision으로 추적할 수 있는가?

## Related

- [[ai/index|AI]]
- [[ai/systems|Systems]]
- [[infra/index|Infra]]
- [[papers/systems/index|Systems papers]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/research-methodology/index|Research methodology]]
- [[agents/verification/verification-loop|Verification loop]]
