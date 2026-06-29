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

## System Contract

AI system은 model weight 하나가 아니라 실행 가능한 contract입니다.

$$
\text{system}
=
(\text{model},\ \text{data},\ \text{code},\ \text{config},\ \text{environment},\ \text{runtime policy})
$$

이 중 하나라도 바뀌면 같은 architecture라도 다른 system으로 취급해야 합니다.

| Contract part | Ask | Start |
| --- | --- | --- |
| data | 어떤 snapshot, split, preprocessing을 썼는가? | [Data validation](/concepts/systems/data-validation), [Dataset split contract](/concepts/data/dataset-split-contract) |
| model | 어떤 weights, tokenizer, featurizer, config인가? | [Model versioning](/concepts/systems/model-versioning), [Model card](/concepts/systems/model-card) |
| training state | checkpoint가 optimizer, scheduler, step, seed를 보존하는가? | [Training run](/concepts/systems/training-run), [Checkpoint state](/concepts/systems/checkpoint-state) |
| inference contract | input/output schema, batching, limits, failure format은 무엇인가? | [Inference contract](/concepts/systems/inference-contract), [Inference](/concepts/systems/inference) |
| environment | dependency, CUDA/runtime, module/container가 고정되는가? | [Environment management](/concepts/systems/environment-management), [Environment modules and containers](/concepts/systems/environment-modules-containers) |
| evidence | 나중에 claim을 재구성할 artifact가 있는가? | [Run artifact](/concepts/systems/run-artifact), [Reproducibility](/concepts/systems/reproducibility) |

## Lifecycle Map

| Stage | Main system risk | Evidence to keep |
| --- | --- | --- |
| data preparation | preprocessing drift or leakage | manifest, checksum, split rule, preprocessing code |
| training | unrecoverable or non-comparable run | config, seed, commit, checkpoint, optimizer state |
| evaluation | metric claim and selection rule mixed | evaluation protocol, baseline, confidence interval |
| packaging | missing tokenizer/featurizer/config | model card, version tag, artifact bundle |
| inference | different preprocessing or batch policy | inference contract, example IO, limits |
| serving | latency/throughput/capacity mismatch | load test, monitoring, rollout record |
| failure recovery | cannot explain or resume a failed run | logs, error class, recovery action, run record |

## Runtime Questions

Many system bugs are caused by asking the wrong layer to explain a symptom.

| Symptom | First route | Then route |
| --- | --- | --- |
| training metric changes after resume | [Checkpoint state](/concepts/systems/checkpoint-state) | [Reproducible run record](/infra/reproducibility/run-record) |
| offline metric is good but service output differs | [Inference contract](/concepts/systems/inference-contract) | [Model serving](/concepts/systems/model-serving) |
| GPU is underutilized | [Memory-compute tradeoff](/concepts/systems/memory-compute-tradeoff) | [GPU](/infra/gpu), [Storage and IO](/infra/io) |
| run cannot be reproduced | [Reproducibility](/concepts/systems/reproducibility) | [Environment management](/concepts/systems/environment-management) |
| scaling claim is unclear | [Scaling claim contract](/concepts/systems/scaling-claim-contract) | [Distributed training](/concepts/systems/distributed-training) |
| service latency tail is high | [Latency and throughput](/concepts/systems/latency-throughput) | [Inference capacity planning](/concepts/systems/inference-capacity-planning) |
| output schema changes across versions | [Model versioning](/concepts/systems/model-versioning) | [Inference contract](/concepts/systems/inference-contract) |

## Systems vs Infra

| 질문 | Systems에서 볼 것 | Infra에서 볼 것 |
| --- | --- | --- |
| 모델을 어떻게 실행 단위로 만들까? | [Training run](/concepts/systems/training-run), [Inference](/concepts/systems/inference) | [HPC](/infra/hpc), [GPU](/infra/gpu) |
| serving capacity를 어떻게 잡을까? | [Inference capacity planning](/concepts/systems/inference-capacity-planning), [Latency and throughput](/concepts/systems/latency-throughput) | [GPU memory](/infra/gpu), [Hardware](/infra/hardware) |
| 결과를 나중에 검증할 수 있을까? | [Run artifact](/concepts/systems/run-artifact), [Experiment lifecycle](/concepts/systems/experiment-lifecycle) | [Reproducibility](/infra/reproducibility) |
| environment 문제가 재현성에 영향을 주나? | [Environment management](/concepts/systems/environment-management) | [Server operations](/infra/server-ops), [HPC](/infra/hpc) |
| bottleneck이 어디인가? | [Memory-compute tradeoff](/concepts/systems/memory-compute-tradeoff), [Storage and IO](/concepts/systems/storage-io) | [GPU](/infra/gpu), [Storage and IO](/infra/io) |

## Claim Types

| Claim | Needs |
| --- | --- |
| model quality | evaluation protocol, split, metric, baseline, uncertainty |
| faster training | matched model/data/quality target, wall time, hardware class, precision |
| cheaper inference | latency, throughput, batch size, memory, hardware class, cache policy |
| reproducible result | run artifact, environment, seed, dataset version, code commit |
| deployable service | inference contract, monitoring, failure mode, rollout and rollback |
| scalable system | scaling curve, communication cost, data loading, resource request |

Do not mix these claims. A better metric does not prove a better service, and a faster service does not prove a better model.

## Public Boundary

Systems notes often sit close to private runs. Public pages should keep reusable contracts and remove operational details.

| Keep | Remove |
| --- | --- |
| hardware class | real hostname, node name, IP, SSH port |
| generic path role | private absolute path |
| environment type | internal module tree or registry |
| rounded runtime or qualitative bottleneck | unpublished experiment result |
| public dataset version | private dataset path or collaborator detail |
| command pattern | credentials, tokens, live endpoint |

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
