---
title: Structured Prediction
tags:
  - tasks
  - structured-prediction
  - machine-learning
---

# Structured Prediction

Structured prediction outputs objects with internal constraints, not just independent labels. Examples include sequences, trees, graphs, segmentations, alignments, molecule strings, tool-call traces, and coordinate sets.

The model predicts:

$$
\hat{y}
=
\arg\max_{y\in\mathcal{Y}(x)}
s_\theta(x,y)
$$

where $\mathcal{Y}(x)$ is the valid output space and $s_\theta$ scores candidate structures.

For probabilistic models:

$$
p_\theta(y\mid x)
=
\prod_{t=1}^{T}
p_\theta(y_t\mid y_{<t},x)
$$

when the structure is generated autoregressively.

## Key Ideas

- The output must satisfy syntax, topology, geometry, or task-specific constraints.
- Decoding is part of the model behavior; greedy, beam, constrained, and sampling-based decoding can produce different outcomes.
- Token-level accuracy may not match object-level validity.
- Structured outputs often need task-specific metrics, validity checks, and error analysis.

## Practical Checks

- What makes an output valid?
- Is the output sequence, set, graph, mask, alignment, coordinate object, or action trace?
- Does the metric evaluate local pieces or the whole object?
- Are invalid outputs filtered, repaired, penalized, or counted as failures?
- Does decoding leak information from references or evaluation tools?
- Is the validity function specified before looking at predictions?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/evaluation/metric|Metric]]
