---
title: Pooling and Readout
tags:
  - architectures
  - readout
  - representation-learning
---

# Pooling and Readout

Pooling and readout layers convert token, node, patch, or residue states into task-level outputs. They determine how local representations become sequence-level, graph-level, or complex-level predictions.

Mean pooling is:

$$
h_{\mathrm{pool}}
= \frac{1}{n}\sum_{i=1}^{n} h_i
$$

Attention pooling uses learned weights:

$$
\alpha_i
= \frac{\exp(a^\top h_i)}
\sum_{j=1}^{n}\exp(a^\top h_j)}
$$

$$
h_{\mathrm{pool}} = \sum_{i=1}^{n}\alpha_i h_i
$$

For padded batches, pooling must use a mask:

$$
h_{\mathrm{mean}}
=
\frac{\sum_i m_i h_i}{\sum_i m_i}
$$

where $m_i=1$ for valid tokens and $0$ for padding. Without this, longer padding or special tokens change the representation.

Task heads then map the readout into the output space:

$$
\hat{y} = g_\phi(h_{\mathrm{pool}})
$$

For graph-level prediction, sum pooling can preserve extensive quantities while mean pooling is better for size-normalized properties.

## Common Choices

| Choice | Typical Use | Risk |
| --- | --- | --- |
| last-token readout | causal sequence models | last position may not summarize bidirectional context |
| CLS-token readout | encoder-style Transformers | CLS behavior depends on pretraining objective |
| mean pooling | size-normalized sequence, set, graph embeddings | rare active sites can be diluted |
| sum pooling | extensive graph properties or counts | larger objects can dominate scale |
| max pooling | detect presence of strong local signal | ignores frequency and context |
| attention pooling | variable-size objects with learned importance | attention weights are not automatically explanations |
| pair readout | interaction, retrieval, matching, protein-ligand scoring | pair construction and negatives define the task |
| task-specific head | node, edge, graph, coordinate, or ranking output | head capacity can dominate representation claim |

## Output Unit Map

| Output Unit | Readout Boundary |
| --- | --- |
| token/node/residue | no global pooling before prediction; output stays equivariant to element order |
| edge/pair | combine two endpoint states plus edge/pair features |
| graph/molecule/protein | permutation-invariant pooling over all valid elements |
| pocket or local site | pool only the local region with a defined mask |
| protein-ligand complex | separate protein, ligand, and cross-interaction readouts when claim needs attribution |
| generated coordinates | readout should preserve vector/coordinate equivariance until the coordinate head |

## Extensive vs Intensive Targets

Pooling should match the target scale:

$$
y_{\mathrm{sum}} \approx \sum_i y_i
\qquad\text{vs}\qquad
y_{\mathrm{mean}} \approx \frac{1}{n}\sum_i y_i
$$

Use sum-like readout for extensive quantities that grow with object size, and mean/normalized readout for size-independent properties. If the target is a binding score, activity label, or class probability, the correct readout depends on how the label was measured.

## Checks

- Does the task need token-level, node-level, graph-level, or global output?
- Does pooling preserve important rare sites or dilute them?
- Are padding tokens excluded from the pool?
- Is the readout invariant to permutations when it should be?
- Is the readout fitted, selected, or tuned on validation data only?
- Does the readout leak future information through masks, pockets, or labels?
- Is the same readout used for baselines when claiming representation quality?

## Related

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/tasks/property-prediction|Property prediction]]
