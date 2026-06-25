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
- [[concepts/geometric-deep-learning/invariance|Invariance]]
