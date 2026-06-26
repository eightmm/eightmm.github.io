---
title: Deep Sets
tags:
  - architectures
  - set-modeling
---

# Deep Sets

Deep Sets are architectures for unordered sets. They are useful when input order should not change the output, such as sets of points, molecules in a batch, retrieved candidates, or unordered observations.

A permutation-invariant set function can be written as:

$$
f(X) = \rho\left(\sum_{x_i\in X}\phi(x_i)\right)
$$

Here $\phi$ embeds each element and $\rho$ maps the pooled representation to the output.

Permutation invariance means:

$$
f(\{x_1,\ldots,x_n\})
=
f(\{x_{\pi(1)},\ldots,x_{\pi(n)}\})
$$

for any permutation $\pi$. If the output is one value per element, the model should be permutation-equivariant instead:

$$
g(\pi X) = \pi g(X)
$$

## Design Choices

- Sum pooling preserves set size information; mean pooling removes direct size scale.
- Max pooling focuses on the strongest element but can ignore aggregate evidence.
- Attention pooling is more expressive but must still be designed to respect permutation behavior.
- Independent $\phi(x_i)$ embeddings miss pairwise interactions unless $\rho$ can infer them from aggregates or additional features.

Deep Sets are often the simplest baseline before using [[concepts/architectures/set-transformer|Set Transformer]], [[concepts/architectures/gnn|Graph neural networks]], or cross-attention over candidates.

## Why It Matters

- Encodes permutation invariance by construction.
- Provides a baseline before using attention or graph structure.
- Clarifies the difference between set inputs and sequence inputs.

## Checks

- Should the output be invariant or equivariant to permutation?
- Is sum, mean, max, or attention pooling used?
- Does the model need pairwise interactions beyond independent element embeddings?

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
