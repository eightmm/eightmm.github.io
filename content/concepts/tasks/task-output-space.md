---
title: Task Output Space
tags:
  - tasks
  - evaluation
  - machine-learning
---

# Task Output Space

A task output space defines what values a model is allowed to produce. It is the difference between predicting a class, a scalar, a ranked list, a sequence, a mask, a box, a graph, coordinates, an answer, or an action.

A task can be written as:

$$
f_\theta : \mathcal{X} \rightarrow \mathcal{Y}
$$

where $\mathcal{X}$ is the input space and $\mathcal{Y}$ is the output space.

If only some outputs are valid for an input $x$, define:

$$
\mathcal{Y}_{\mathrm{valid}}(x)
=
\{y\in\mathcal{Y}: v(x,y)=1\}
$$

where $v$ is the validity function.

## Common Output Spaces

- Class label: $y\in\{1,\ldots,K\}$.
- Scalar or vector: $y\in\mathbb{R}^d$.
- Ranked list: $y=(d_1,\ldots,d_k)$ with ordered candidates.
- Retrieved set: $y\subseteq\mathcal{D}$.
- Sequence: $y=(y_1,\ldots,y_T)$ with syntax or vocabulary constraints.
- Spatial object: box, mask, keypoints, segmentation map, or dense field.
- Graph: nodes, edges, labels, or generated graph structure.
- Coordinates: $y=(x_1,\ldots,x_N)$ with geometric constraints.
- Action: tool call, policy action, plan step, or control command.

## Loss and Metric Implications

Changing $\mathcal{Y}$ changes the loss and metric:

$$
\mathcal{Y}
\Rightarrow
(\mathcal{L}, \mathcal{M}, v)
$$

where $\mathcal{L}$ is the training loss, $\mathcal{M}$ is the metric set, and $v$ is the validity check.

For example, a generated molecule string can be scored by token loss, validity, uniqueness, property prediction, docking score, or downstream assay relevance. These are not interchangeable.

The output space should be interpreted together with [[concepts/modalities/modality-task-map|Modality-task map]], because the same raw modality can produce a class, scalar, ranking, sequence, mask, graph, coordinates, or action depending on the task.

Changing the output space usually changes [[concepts/evaluation/metric-selection|Metric selection]] and the relevant [[concepts/evaluation/failure-mode-taxonomy|failure modes]].

## Checks

- What exactly is one target output?
- Is the output independent, ranked, sequential, spatial, temporal, geometric, graph-structured, or an action?
- What makes an output invalid?
- Are invalid outputs filtered, repaired, penalized, or counted as failures?
- Does the metric score local pieces, whole-object validity, utility, or human preference?
- Does decoding or post-processing change the effective output space?

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
