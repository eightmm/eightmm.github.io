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

More explicitly:

$$
\|A-A_k\|_F
=
\min_{\operatorname{rank}(B)\le k}
\|A-B\|_F
$$

The approximation error is:

$$
\|A-A_k\|_F^2
=
\sum_{i>k}
\sigma_i^2
$$

where $\sigma_i$ are singular values sorted in descending order.

## Relation to Eigenvalues

SVD is connected to eigen-decomposition:

$$
A^\top A
=
V\Sigma^\top\Sigma V^\top
$$

The squared singular values are eigenvalues of $A^\top A$:

$$
\lambda_i(A^\top A)
=
\sigma_i^2
$$

This is why singular values measure how strongly a linear map stretches directions, even when $A$ is not square.

## Representation Diagnostics

For an embedding matrix $Z\in\mathbb{R}^{n\times d}$, singular values can reveal whether information is spread across dimensions:

$$
Z
=
U\Sigma V^\top
$$

A rapid collapse of $\sigma_i$ may indicate redundant dimensions or representation collapse. This is a diagnostic, not proof by itself.

## Why It Matters

- PCA can be implemented with SVD on centered data.
- Embedding matrices can be inspected through singular values.
- Low-rank adapters and factorized layers use similar structure.
- Representation collapse can appear as a sharp drop in singular values.

## Checks

- Is the matrix centered or normalized before applying SVD?
- Are singular values used for compression, diagnostics, PCA, or numerical stability?
- Does a low-rank approximation preserve the task-relevant signal?
- Are small singular values numerical noise or meaningful low-variance directions?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/learning/representation-collapse|Representation collapse]]
