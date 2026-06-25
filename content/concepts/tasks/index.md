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

- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/tasks/question-answering|Question answering]]
- [[concepts/tasks/sequence-generation|Sequence generation]]

## Checks

- What is the output space?
- Is the model predicting, ranking, retrieving, generating, localizing, or segmenting?
- Does the metric match the user-facing behavior?
- Does the data split prevent leakage for this task?
- Are outputs constrained to valid objects, syntax, molecules, structures, or actions?

## Related

- [[ai/index|AI]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/evaluation/index|Evaluation]]
