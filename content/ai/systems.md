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

AI system note의 핵심은 “무슨 모델을 썼다”가 아니라 어떤 artifact가 어떤 contract로 재현되고 실행되는지입니다. 같은 model weight라도 dataset snapshot, tokenizer, preprocessing, precision, batching, serving policy가 달라지면 다른 system입니다.

## System Contract

AI system은 모델 파일 하나가 아니라, 입력에서 artifact와 claim까지 이어지는 실행 계약입니다.

$$
S
=
(D,\ f_\theta,\ C,\ E,\ R,\ A,\ V)
$$

| Part | Meaning | Example |
| --- | --- | --- |
| $D$ | data contract | dataset snapshot, split, preprocessing, feature schema |
| $f_\theta$ | model artifact | architecture, weights, tokenizer or featurizer |
| $C$ | config | hyperparameters, precision, context length, batching rule |
| $E$ | environment | package set, container/module, CUDA/runtime boundary |
| $R$ | runtime policy | training loop, inference mode, serving policy, scheduler request |
| $A$ | artifacts | checkpoint, logs, metrics, generated outputs, report |
| $V$ | verification | tests, metrics, rendered outputs, reproducibility evidence |

This gives a practical rule:

$$
\text{same weights}
\neq
\text{same system}
$$

if data preprocessing, tokenizer, environment, precision, batching, or serving policy changed.

## Main Areas

| Area | Use for | Key notes |
| --- | --- | --- |
| Training | run state, checkpoint, scaling, failure recovery | [Training run](/concepts/systems/training-run), [Checkpoint state](/concepts/systems/checkpoint-state), [Distributed training](/concepts/systems/distributed-training) |
| Inference | prediction-time execution, batch/online mode, output contract | [Inference](/concepts/systems/inference), [Batch and online inference](/concepts/systems/batch-online-inference), [Inference contract](/concepts/systems/inference-contract) |
| Serving | endpoint, batching, capacity, latency, rollout | [Model serving](/concepts/systems/model-serving), [Inference serving](/concepts/systems/inference-serving), [Inference capacity planning](/concepts/systems/inference-capacity-planning), [Latency and throughput](/concepts/systems/latency-throughput) |
| Environment | module, container, dependency, runtime drift | [Environment management](/concepts/systems/environment-management), [Environment modules and containers](/concepts/systems/environment-modules-containers) |
| Reproducibility | run record, artifact, versioning, evidence boundary | [Reproducibility](/concepts/systems/reproducibility), [Run artifact](/concepts/systems/run-artifact), [Model versioning](/concepts/systems/model-versioning) |
| Operations | monitoring, deployment, recovery, resource scheduling | [Deployment strategy](/concepts/systems/deployment-strategy), [Observability](/concepts/systems/observability), [Failure recovery](/concepts/systems/failure-recovery), [Resource scheduling](/concepts/systems/resource-scheduling) |

## Route by Object

| If you are describing | Start here | Then route to |
| --- | --- | --- |
| model structure or inductive bias | [[ai/architectures|Architectures]] | systems only after it becomes a runnable artifact |
| objective, optimizer, supervision, transfer | [[ai/machine-learning|Machine Learning]], [[ai/learning-methods|Learning Methods]] | systems for run record and checkpoint contract |
| evaluation metric or benchmark claim | [[ai/evaluation|Evaluation]] | systems for artifact and reproducibility boundary |
| serving endpoint or batch inference | this page | [[concepts/systems/inference-contract|Inference contract]], [[concepts/systems/model-serving|Model serving]] |
| GPU, Slurm, storage, account, server issue | [[infra/index|Infra]] | systems only for run/artifact implications |
| paper or project implementation | [[papers/index|Papers]], [[projects/index|Projects]] | systems for reproducible artifact description |

## Lifecycle

| Stage | System question | Evidence |
| --- | --- | --- |
| Data preparation | 어떤 snapshot과 preprocessing contract를 썼는가? | manifest, split, script, checksum |
| Training | 어떤 run state가 checkpoint를 만들었는가? | config, seed, optimizer, logs |
| Evaluation | 어떤 metric과 selection rule로 claim을 만들었는가? | eval table, baseline, confidence interval |
| Inference | input/output contract가 무엇인가? | schema, example, batch policy |
| Serving | latency, throughput, capacity boundary는 무엇인가? | benchmark, monitoring, rollout record |
| Reproducibility | 같은 claim을 다시 재구성할 수 있는가? | run artifact, version tag, environment |

## Boundary With Infra

| Question | Route |
| --- | --- |
| training 또는 inference concept인가? | [AI Systems](/ai/systems), [Systems concepts](/concepts/systems) |
| hardware, GPU, Slurm, storage, server operation이 관련되는가? | [Infra](/infra) |
| run 또는 artifact가 reproducible하다는 public record가 필요한가? | [Reproducibility](/concepts/systems/reproducibility), [Run record](/infra/reproducibility/run-record) |
| cluster 또는 shared machine에서 실패했는가? | [HPC](/infra/hpc), [GPU](/infra/gpu), [Server operations](/infra/server-ops) |

## Boundary Examples

| Note topic | Put in Systems | Put in Infra |
| --- | --- | --- |
| training run | checkpoint contents, config, seed, resume contract | Slurm request, GPU allocation, shared filesystem bottleneck |
| inference | input/output schema, batching rule, output validation | GPU memory, server process, network, deployment machine |
| environment | dependency contract and reproducibility metadata | module path, container runtime issue, driver/runtime mismatch |
| serving | latency/throughput target, rollout, fallback behavior | node capacity, monitoring setup, firewall, process supervision |
| artifact | what proves the claim can be reconstructed | where artifacts are stored and backed up |

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

## Claim Pattern

For public writing, reduce a systems claim to:

$$
(\text{data},\ \text{model},\ \text{config},\ \text{environment},\ \text{runtime},\ \text{artifact},\ \text{verifier})
\rightarrow
\text{claim}
$$

Examples:

| Claim | Required evidence |
| --- | --- |
| training is reproducible | data snapshot, code revision, config, seed, environment, checkpoint, metric record |
| inference is stable | input schema, output schema, batch policy, error handling, validation examples |
| serving meets latency target | workload shape, hardware class, p50/p95/p99, concurrency, timeout policy |
| checkpoint can resume | optimizer, scheduler, scaler, RNG, step, data position, artifact integrity |
| result can be published | public-safe artifact, metric protocol, baseline, sanitization boundary |

## System Failure Modes

- checkpoint는 남았지만 tokenizer, featurizer, preprocessing rule이 없습니다.
- evaluation metric은 있지만 split, selection rule, confidence interval이 없습니다.
- local inference는 되지만 serving batch/precision/timeout 조건이 다릅니다.
- hardware 문제를 model quality 문제로 오해합니다.
- deployment는 성공했지만 public claim을 뒷받침하는 run record가 없습니다.
- inference example은 있지만 input/output schema와 error contract가 없습니다.
- serving benchmark는 있지만 request distribution과 concurrency assumption이 없습니다.
- reproducibility note가 private paths, server names, or unpublished results에 의존합니다.

## Checks

- model, data, code, config, environment, hardware class가 분리되어 기록됐는가?
- training artifact와 inference artifact가 같은 preprocessing contract를 쓰는가?
- metric claim과 serving requirement가 서로 다른 evidence를 요구한다는 점을 구분했는가?
- public note에 private path, server name, account, unpublished result가 들어가지 않았는가?
- artifact가 “생겼다”와 claim이 “검증됐다”를 구분했는가?
- Infra incident를 model/system design claim으로 과도하게 일반화하지 않았는가?

## Related

- [[ai/index|AI]]
- [[concepts/systems/index|AI systems concepts]]
- [[infra/index|Infra]]
- [[projects/index|Projects]]
