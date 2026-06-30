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

## Spectral Norm and Conditioning

The largest singular value is the spectral norm:

$$
\lVert A\rVert_2=\sigma_{\max}(A)
$$

For a full-rank square matrix, the condition number is:

$$
\kappa(A)
=
\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

A large condition number means some directions are amplified much more than others. This can make optimization, linear solves, and representation inversion numerically fragile.

## PCA Connection

For centered data matrix $X_c\in\mathbb{R}^{n\times d}$:

$$
X_c = U\Sigma V^\top
$$

The covariance matrix is:

$$
\hat{\Sigma}
=
\frac{1}{n-1}X_c^\top X_c
=
V\frac{\Sigma^\top\Sigma}{n-1}V^\top
$$

Thus PCA directions are right singular vectors of centered data, and explained variances are:

$$
\lambda_i
=
\frac{\sigma_i^2}{n-1}
$$

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
- Is the matrix raw, centered, normalized, whitened, or batched?
- Is the conclusion based on singular values alone or validated by downstream behavior?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/learning/representation-collapse|Representation collapse]]
