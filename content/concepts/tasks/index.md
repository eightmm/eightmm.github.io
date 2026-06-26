---
title: Tasks
tags:
  - tasks
  - machine-learning
---

# Tasks

A task defines what the model must output and how success is measured. Architecture describes how information flows; modality describes the input signal; task describes the target behavior.

The same input can support many tasks:

$$
\hat{y} = f_\theta(x), \qquad \theta^\star = \arg\min_\theta \mathcal{L}(f_\theta(x), y)
$$

The important question is what $y$ means: a class, scalar, ranked list, generated sequence, box, mask, answer, retrieved item, or structured object.

## Route Map

| Need | Start | Typical Metric Boundary |
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

## Task Families

| Family | Notes |
| --- | --- |
| Core ML | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking) |
| Search | [Retrieval](/concepts/tasks/retrieval), [Similarity search](/concepts/tasks/similarity-search), [Reranking](/concepts/tasks/reranking) |
| Vision and spatial | [Object detection](/concepts/tasks/object-detection), [Localization](/concepts/tasks/localization), [Segmentation](/concepts/tasks/segmentation) |
| Language | [Captioning](/concepts/tasks/captioning), [Question answering](/concepts/tasks/question-answering), [Sequence generation](/concepts/tasks/sequence-generation) |
| Structured outputs | [Structured prediction](/concepts/tasks/structured-prediction), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Time and monitoring | [Time-series forecasting](/concepts/tasks/time-series-forecasting), [Anomaly detection](/concepts/tasks/anomaly-detection) |

## Checks

- What is the output space?
- What [[concepts/tasks/task-output-space|task output space]] constrains valid predictions?
- What is the task specification: input, output, validity, loss, metric, and split?
- What modality-task map connects raw input, representation, output space, and metric?
- What is the [[concepts/data/example-unit|example unit]]?
- What [[concepts/data/split-unit|split unit]] tests the intended generalization?
- Is the model predicting, ranking, retrieving, generating, localizing, or segmenting?
- Is retrieval a final output, a similarity-search stage, or a reranking pipeline?
- Is the task entity-level, pairwise, context-conditioned, or structured?
- Is the output independent, sequential, structured, ranked, spatial, or temporal?
- Does the metric match the user-facing behavior?
- Does the data split prevent leakage for this task?
- Are outputs constrained to valid objects, syntax, molecules, structures, or actions?

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
