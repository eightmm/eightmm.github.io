---
title: Task Specification
tags:
  - tasks
  - evaluation
  - machine-learning
---

# Task Specification

A task specification defines what a model receives, what it must output, what outputs are valid, and how success is measured. It is the contract between data, model, and evaluation.

A compact task specification is:

$$
\mathcal{T}
=
(\mathcal{X}, \mathcal{Y}, v, \mathcal{L}, \mathcal{M}, s)
$$

where $\mathcal{X}$ is the input space, $\mathcal{Y}$ is the [[concepts/tasks/task-output-space|output space]], $v$ is a validity function, $\mathcal{L}$ is the training loss, $\mathcal{M}$ is the metric set, and $s$ is the split rule.

## Required Fields

- Input: raw modality and model-ready representation.
- Output: class, scalar, rank, retrieved set, sequence, mask, box, graph, coordinate object, answer, or action.
- Validity: syntax, geometry, schema, label space, constraints, or safety boundary.
- Loss: what is optimized during training.
- Metric: what is reported during evaluation.
- Split: what generalization claim the result supports.
- Failure mode: invalid output, wrong output, unsupported output, missing evidence, or low-confidence output.

## Why It Matters

Two systems can use the same architecture but solve different tasks. A decoder-only transformer can do sequence generation, retrieval reranking, tool-call structured output, or classification depending on the task specification.

The task should also be checked against [[concepts/modalities/modality-task-map|Modality-task map]] so the raw modality, representation, output space, loss, metric, and split rule stay aligned.

## Checks

- What is one example and one target?
- Is the output independent, ranked, sequential, spatial, temporal, or structured?
- Are invalid outputs counted as failures or repaired before scoring?
- Does the metric measure the actual downstream behavior?
- Does the split unit match the claimed generalization?
- Is the training loss aligned with the evaluation metric?

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/modalities/index|Modalities]]
