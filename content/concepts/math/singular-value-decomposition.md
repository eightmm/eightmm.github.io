---
title: Singular Value Decomposition
tags:
  - math
  - linear-algebra
---

# Singular Value Decomposition

Singular value decomposition decomposes any matrix into input directions, output directions, and scaling factors.

For $A \in \mathbb{R}^{m\times n}$:

$$
A = U\Sigma V^\top
$$

where:

- $U \in \mathbb{R}^{m\times m}$ contains left singular vectors.
- $V \in \mathbb{R}^{n\times n}$ contains right singular vectors.
- $\Sigma \in \mathbb{R}^{m\times n}$ is diagonal or rectangular-diagonal with nonnegative singular values.

## Interpretation

The matrix first rotates or reflects by $V^\top$, scales by $\Sigma$, then rotates or reflects by $U$.

For an input vector $x$:

$$
Ax = U\Sigma V^\top x
$$

Large singular values identify directions preserved or amplified by the transformation.

## Low-Rank Approximation

The rank-$k$ approximation keeps the largest $k$ singular values:

$$
A_k = U_k\Sigma_k V_k^\top
$$

This is the best rank-$k$ approximation under the Frobenius norm.

## Why It Matters

- PCA can be implemented with SVD on centered data.
- Embedding matrices can be inspected through singular values.
- Low-rank adapters and factorized layers use similar structure.
- Representation collapse can appear as a sharp drop in singular values.

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
