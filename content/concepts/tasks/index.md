---
title: Tasks
tags:
  - tasks
  - machine-learning
---

# Tasks

Task는 model이 무엇을 출력해야 하고 성공을 어떻게 측정할지 정의합니다. Architecture는 정보가 흐르는 방식을, modality는 input signal을, task는 target behavior를 설명합니다.

같은 input도 여러 task를 가질 수 있습니다.

$$
\hat{y} = f_\theta(x), \qquad \theta^\star = \arg\min_\theta \mathcal{L}(f_\theta(x), y)
$$

중요한 질문은 $y$가 무엇을 뜻하는가입니다. $y$는 class, scalar, ranked list, generated sequence, box, mask, answer, retrieved item, structured object일 수 있습니다.

## 이동 지도

| 필요 | 시작점 | 일반적인 metric 경계 |
| --- | --- | --- |
| define the full task contract | [Task specification](/concepts/tasks/task-specification) | loss, metric, split, validity rule |
| choose the output type | [Task output space](/concepts/tasks/task-output-space) | class, scalar, rank, sequence, graph, coordinate |
| connect input modality to task | [Modality-task map](/concepts/modalities/modality-task-map) | raw input, representation, output, metric |
| predict labels or values | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression) | calibration, threshold, error scale |
| rank or retrieve items | [Ranking](/concepts/machine-learning/ranking), [Retrieval](/concepts/tasks/retrieval), [Reranking](/concepts/tasks/reranking) | top-k, NDCG, enrichment, recall |
| generate objects | [Sequence generation](/concepts/tasks/sequence-generation), [Structured prediction](/concepts/tasks/structured-prediction) | validity, diversity, likelihood, downstream score |
| localize in space | [Localization](/concepts/tasks/localization), [Object detection](/concepts/tasks/object-detection), [Segmentation](/concepts/tasks/segmentation) | IoU, mask quality, localization error |
| model molecules or proteins | [Property prediction](/concepts/tasks/property-prediction), [Interaction prediction](/concepts/tasks/interaction-prediction) | assay context, split unit, leakage risk |
| model structure | [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) | geometry validity, invariance, equivariance |

## Task 묶음

| 묶음 | 노트 |
| --- | --- |
| Core ML | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking) |
| Search | [Retrieval](/concepts/tasks/retrieval), [Similarity search](/concepts/tasks/similarity-search), [Reranking](/concepts/tasks/reranking) |
| Vision and spatial | [Object detection](/concepts/tasks/object-detection), [Localization](/concepts/tasks/localization), [Segmentation](/concepts/tasks/segmentation) |
| Language | [Captioning](/concepts/tasks/captioning), [Question answering](/concepts/tasks/question-answering), [Sequence generation](/concepts/tasks/sequence-generation) |
| Structured outputs | [Structured prediction](/concepts/tasks/structured-prediction), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Time and monitoring | [Time-series forecasting](/concepts/tasks/time-series-forecasting), [Anomaly detection](/concepts/tasks/anomaly-detection) |

## 확인할 것

- output space는 무엇인가?
- 어떤 [[concepts/tasks/task-output-space|task output space]]가 valid prediction을 제한하는가?
- input, output, validity, loss, metric, split을 포함한 task specification이 있는가?
- raw input, representation, output space, metric을 연결하는 modality-task map이 있는가?
- [[concepts/data/example-unit|example unit]]은 무엇인가?
- 어떤 [[concepts/data/split-unit|split unit]]이 의도한 generalization을 검증하는가?
- model은 predicting, ranking, retrieving, generating, localizing, segmenting 중 무엇을 하는가?
- retrieval은 최종 output인가, similarity-search stage인가, reranking pipeline인가?
- task는 entity-level, pairwise, context-conditioned, structured 중 무엇인가?
- output은 independent, sequential, structured, ranked, spatial, temporal 중 무엇인가?
- metric이 user-facing behavior와 맞는가?
- data split이 이 task의 leakage를 막는가?
- output이 valid object, syntax, molecule, structure, action으로 제한되는가?

## Related

- [[ai/index|AI]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
