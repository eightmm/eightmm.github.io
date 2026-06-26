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

- Last-token readout for causal sequence models.
- CLS-token readout for encoder-style Transformers.
- Mean or sum pooling for sets and graphs.
- Attention pooling for variable-size objects.
- Task-specific heads for node-level, edge-level, or graph-level prediction.

## Checks

- Does the task need token-level, node-level, graph-level, or global output?
- Does pooling preserve important rare sites or dilute them?
- Are padding tokens excluded from the pool?
- Is the readout invariant to permutations when it should be?

## Related

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/tasks/property-prediction|Property prediction]]
