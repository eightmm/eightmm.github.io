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

## Task Families

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/tasks/question-answering|Question answering]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/tasks/time-series-forecasting|Time-series forecasting]]
- [[concepts/tasks/anomaly-detection|Anomaly detection]]

## Checks

- What is the output space?
- What [[concepts/tasks/task-output-space|task output space]] constrains valid predictions?
- What is the task specification: input, output, validity, loss, metric, and split?
- What modality-task map connects raw input, representation, output space, and metric?
- What is the [[concepts/data/example-unit|example unit]]?
- What [[concepts/data/split-unit|split unit]] tests the intended generalization?
- Is the model predicting, ranking, retrieving, generating, localizing, or segmenting?
- Is retrieval a final output, a similarity-search stage, or a reranking pipeline?
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
