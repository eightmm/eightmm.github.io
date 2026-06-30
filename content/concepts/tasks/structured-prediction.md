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

where $\mathcal{Y}(x)$ is the valid [[concepts/tasks/task-output-space|output space]] and $s_\theta$ scores candidate structures.

For probabilistic models:

$$
p_\theta(y\mid x)
=
\prod_{t=1}^{T}
p_\theta(y_t\mid y_{<t},x)
$$

when the structure is generated autoregressively.

## Output Constraint

Structured prediction has a validity set:

$$
\mathcal{Y}_{\mathrm{valid}}(x)
\subseteq
\mathcal{Y}
$$

The decoder should either search only inside the valid set or report invalid outputs explicitly.

| Output type | Constraint example |
| --- | --- |
| sequence | grammar, vocabulary, length, stop token |
| segmentation mask | pixel/voxel labels and topology |
| graph | node/edge type rules, connectivity, valence |
| coordinates | rigid geometry, distance, chirality, collision |
| tool trace | schema, permissions, side-effect order |
| alignment | monotonicity or one-to-one matching |

## Decoding and Loss

Training often optimizes local losses, while evaluation checks whole-object validity.

$$
\mathcal{L}
=
\sum_t
\ell(y_t,\hat{y}_t)
\quad
\not\Rightarrow
\quad
\hat{y}\in\mathcal{Y}_{\mathrm{valid}}
$$

| Training signal | Evaluation question |
| --- | --- |
| token cross entropy | does the whole sequence parse and solve the task? |
| per-pixel loss | is the object boundary/topology correct? |
| edge classification | is the graph chemically/structurally valid? |
| coordinate loss | are geometry and invariance/equivariance respected? |
| imitation of traces | does the action sequence complete the workflow? |

## Validity Handling

| Strategy | Use when | Risk |
| --- | --- | --- |
| constrained decoding | rules are clear and cheap | may reduce diversity |
| repair step | invalid outputs are near-valid | repair can hide model failure |
| reject invalid | validity is mandatory | lowers yield |
| penalty in objective | validity can be softly measured | may not enforce hard constraints |
| separate verifier | validity check is complex | verifier coverage must be audited |

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
- Is invalid output rate reported separately from quality on valid outputs?
- Does the loss align with object-level success?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/coordinate-prediction|Coordinate prediction]]
- [[concepts/tasks/graph-prediction|Graph prediction]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/evaluation/metric|Metric]]
