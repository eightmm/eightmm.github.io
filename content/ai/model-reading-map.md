---
title: Model Reading Map
tags:
  - ai
  - paper-reading
---

# Model Reading Map

AI 논문을 읽을 때는 모델 이름보다 먼저 입력, 표현, 구조, 학습 신호, 평가 claim을 분리합니다. 이 페이지는 한 편의 논문을 wiki note로 옮길 때 쓰는 절차형 지도입니다.

$$
\text{paper claim}
=
\text{object}
+ \text{representation}
+ \text{architecture}
+ \text{objective}
+ \text{evidence}
$$

논문을 읽을 때 가장 먼저 해야 할 일은 “새 모델 이름”을 외우는 것이 아니라 claim의 단위를 고정하는 것입니다. 같은 architecture라도 input object, split, objective, evaluation이 달라지면 완전히 다른 주장입니다.

## Reading Order

| Step | Ask | Route |
| --- | --- | --- |
| 1. Object | 입력 단위가 sample, token, image, graph, molecule, protein, pocket, agent state 중 무엇인가? | [Modalities](/concepts/modalities), [Entities](/molecular-modeling/entities) |
| 2. Representation | raw object가 token, embedding, graph, coordinate, conformer, feature로 어떻게 바뀌는가? | [Representation contract](/concepts/modalities/representation-contract) |
| 3. Architecture | 어떤 축을 섞고 어떤 inductive bias를 쓰는가? | [Architectures](/ai/architectures), [Architecture-objective fit](/concepts/architectures/architecture-objective-fit) |
| 4. Objective | label, mask, contrast, likelihood, denoising, flow, preference, reward 중 무엇을 최적화하는가? | [Learning Methods](/ai/learning-methods), [Objective taxonomy](/concepts/learning/objective-taxonomy) |
| 5. Training | data source, split, optimizer, schedule, selection rule이 claim과 맞는가? | [Machine Learning](/ai/machine-learning), [Systems](/ai/systems) |
| 6. Evaluation | metric, baseline, uncertainty, leakage check가 주장을 지지하는가? | [Evaluation](/ai/evaluation) |
| 7. Runtime | inference cost, serving boundary, reproducibility가 실제 사용 조건과 맞는가? | [Systems](/ai/systems), [Infra](/infra) |

## Core Symbols

논문 note를 작성할 때는 최소한 아래 기호를 먼저 고정합니다.

| Symbol | Meaning |
| --- | --- |
| $u_i$ | sample unit: molecule, sequence, image, protein-ligand pair, user task |
| $x_i$ | model input after preprocessing |
| $r_i$ | representation: token, graph, coordinate, embedding |
| $y_i$ | label, target, reference, preference, reward |
| $f_\theta$ | model or predictor |
| $\mathcal{L}$ | training objective |
| $\hat{m}$ | finite-sample metric estimate |

## Claim Types

| Claim | 먼저 볼 것 | 흔한 함정 |
| --- | --- | --- |
| Better architecture | 같은 objective와 data에서 architecture만 바뀌었는가? | objective, augmentation, compute 차이가 architecture 효과처럼 보임 |
| Better pretraining | downstream split과 pretraining corpus가 분리되는가? | duplicate, homolog, scaffold, source leakage |
| Better generation | sample quality와 likelihood/score/evaluator가 같은 claim인가? | filtered denominator를 숨김 |
| Better representation | frozen probe, kNN, retrieval, fine-tuning 중 무엇으로 검증했는가? | probe budget이 representation 차이를 가림 |
| Better system | latency, memory, throughput, reproducibility를 같은 조건에서 비교했는가? | hardware, precision, batch size 차이 |

## Reading Pitfalls

- architecture claim인데 실제 차이는 data filtering, augmentation, compute일 수 있습니다.
- representation claim인데 downstream fine-tuning budget이 결과를 지배할 수 있습니다.
- generation claim인데 validity filtering 뒤의 denominator가 빠질 수 있습니다.
- evaluation claim인데 split unit과 deployment unit이 다를 수 있습니다.
- system claim인데 hardware, precision, batch size, caching condition이 다를 수 있습니다.

## Minimal Paper Note

| Field | Write |
| --- | --- |
| Problem | task unit, input $x$, target/output $y$ |
| Representation | token/graph/coordinate/embedding contract |
| Architecture | family, key operation, inductive bias, scaling cost |
| Objective | exact loss or training signal with symbols defined |
| Evidence | split, metric, baseline, uncertainty, ablation |
| Boundary | AI, Math, Computational Biology, Infra, Agents 중 어디로 넘길지 |

## Route Decision

| 내용 | 둘 곳 |
| --- | --- |
| model-internal structure, objective, training/evaluation claim | [AI](/ai) |
| formula shape, estimator, probability, gradient, geometry primitive | [Math](/math) |
| protein, molecule, ligand, pocket, docking, assay, structure workflow | [Computational Biology](/molecular-modeling) |
| GPU, Slurm, storage, deployment environment | [Infra](/infra) |
| tool-using LLM workflow, paper brief automation, wiki maintenance | [Agents](/agents) |

## Related

- [[ai/index|AI]]
- [[ai/architectures|Architectures]]
- [[ai/learning-methods|Learning Methods]]
- [[math/formula-reading|Formula Reading]]
- [[molecular-modeling/representation-routes|Representation Routes]]
