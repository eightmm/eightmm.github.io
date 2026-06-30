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

## Invariance vs Equivariance

Deep Sets are mainly used for permutation-invariant outputs:

$$
f(\pi X)=f(X)
$$

For per-element outputs, the model should be permutation-equivariant:

$$
g(\pi X)=\pi g(X)
$$

An equivariant Deep Sets-style layer can be written as:

$$
h_i
=
\psi\left(x_i,\ \sum_j \phi(x_j)\right)
$$

Each element receives its own feature $x_i$ plus a shared summary of the whole set.

## Pooling Choice

| Pooling | Preserves | Loses or risks |
| --- | --- | --- |
| sum | count and aggregate magnitude | scale grows with set size |
| mean | average property independent of size | count information |
| max | strongest evidence | weak distributed evidence |
| attention pooling | learned importance | may become order-sensitive if implemented incorrectly |

If set size itself matters, mean pooling alone may be insufficient. If only composition matters, sum pooling may introduce unwanted size dependence.

## Expressivity Boundary

The basic form

$$
\rho\left(\sum_i \phi(x_i)\right)
$$

captures interactions only through the pooled summary. It is a strong baseline, but it can be weak when the task depends on pairwise or higher-order relations.

| Needed structure | Better candidate |
| --- | --- |
| pairwise distances or contacts | [[concepts/architectures/gnn|Graph neural networks]] |
| learned interactions among set elements | [[concepts/architectures/set-transformer|Set Transformer]] |
| query-specific reading from a set | [[concepts/architectures/cross-attention|Cross-attention]] |
| geometric equivariance | [[concepts/geometric-deep-learning/equivariance|Equivariance]] |

## Useful Applications

| Input set | Possible output |
| --- | --- |
| retrieved evidence chunks | answer support score or reranked context |
| molecular atoms without bonds | pooled molecular descriptor baseline |
| point cloud | object-level classification |
| candidate poses | best-pose or uncertainty summary |
| multiple observations | aggregate prediction under missing order |

## Why It Matters

- Encodes permutation invariance by construction.
- Provides a baseline before using attention or graph structure.
- Clarifies the difference between set inputs and sequence inputs.

## Checks

- Should the output be invariant or equivariant to permutation?
- Is sum, mean, max, or attention pooling used?
- Does the model need pairwise interactions beyond independent element embeddings?
- Does set size carry signal, or should it be normalized away?
- Is any positional/order information accidentally leaking into the set representation?

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
