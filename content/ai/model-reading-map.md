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

## Claim Types

| Claim | 먼저 볼 것 | 흔한 함정 |
| --- | --- | --- |
| Better architecture | 같은 objective와 data에서 architecture만 바뀌었는가? | objective, augmentation, compute 차이가 architecture 효과처럼 보임 |
| Better pretraining | downstream split과 pretraining corpus가 분리되는가? | duplicate, homolog, scaffold, source leakage |
| Better generation | sample quality와 likelihood/score/evaluator가 같은 claim인가? | filtered denominator를 숨김 |
| Better representation | frozen probe, kNN, retrieval, fine-tuning 중 무엇으로 검증했는가? | probe budget이 representation 차이를 가림 |
| Better system | latency, memory, throughput, reproducibility를 같은 조건에서 비교했는가? | hardware, precision, batch size 차이 |

## Minimal Paper Note

| Field | Write |
| --- | --- |
| Problem | task unit, input $x$, target/output $y$ |
| Representation | token/graph/coordinate/embedding contract |
| Architecture | family, key operation, inductive bias, scaling cost |
| Objective | exact loss or training signal with symbols defined |
| Evidence | split, metric, baseline, uncertainty, ablation |
| Boundary | AI, Math, Computational Biology, Infra, Agents 중 어디로 넘길지 |

## Related

- [[ai/index|AI]]
- [[ai/architectures|Architectures]]
- [[ai/learning-methods|Learning Methods]]
- [[math/formula-reading-for-ai|Formula Reading for AI]]
- [[molecular-modeling/representation-routes|Representation Routes]]
