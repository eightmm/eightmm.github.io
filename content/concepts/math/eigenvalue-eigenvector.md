---
title: Eigenvalue and Eigenvector
tags:
  - math
  - linear-algebra
---

# Eigenvalue and Eigenvector

An eigenvector is a direction that a linear transformation only scales, without rotating into a different direction.

For a square matrix $A \in \mathbb{R}^{d\times d}$:

$$
Av = \lambda v
$$

where $v \ne 0$ is an eigenvector and $\lambda$ is the corresponding eigenvalue.

## Interpretation

- $v$ is a special direction of the transformation.
- $\lambda$ is the scale factor along that direction.
- Large absolute eigenvalues indicate directions that are strongly amplified.
- Negative eigenvalues flip the direction.

If $A$ has eigenvectors that form a basis, the transformation can be decomposed as:

$$
A
=
V\Lambda V^{-1}
$$

where columns of $V$ are eigenvectors and $\Lambda$ contains eigenvalues. This makes repeated application easy to analyze:

$$
A^k
=
V\Lambda^k V^{-1}
$$

Eigenvalues with $|\lambda|>1$ grow under repeated application, while values with $|\lambda|<1$ shrink.

## Why It Matters

- PCA uses eigenvectors of a covariance matrix.
- Optimization stability is shaped by curvature and [[concepts/math/jacobian-hessian|Hessian]] eigenvalues.
- Graph methods use eigenvectors of adjacency or Laplacian matrices.
- Dynamical systems and state-space models often analyze eigenvalues for stability.

## Symmetric Matrices

For a real symmetric matrix $A=A^\top$:

$$
A
=
Q\Lambda Q^\top
$$

where $Q$ is orthogonal and eigenvalues are real. This case appears in covariance matrices, Hessians, kernels, and graph Laplacians.

For a covariance matrix, large eigenvalues correspond to high-variance directions:

$$
\Sigma v_i
=
\lambda_i v_i
$$

This is the basis of PCA-style dimensionality reduction.

## Stability View

Eigenvalues describe repeated application of a linear update:

$$
h_{t+1}=Ah_t
$$

If $A=V\Lambda V^{-1}$, then:

$$
h_t=A^th_0=V\Lambda^tV^{-1}h_0
$$

The spectral radius controls growth:

$$
\rho(A)=\max_i|\lambda_i|
$$

| Condition | Behavior |
| --- | --- |
| $\rho(A)<1$ | perturbations shrink |
| $\rho(A)=1$ | marginal stability depends on structure |
| $\rho(A)>1$ | some directions grow |

This is why eigenvalues appear in optimization stability, recurrent models, and state-space models.

## Curvature View

For a twice-differentiable loss near a point $\theta$:

$$
\mathcal{L}(\theta+\delta)
\approx
\mathcal{L}(\theta)
+
\nabla\mathcal{L}(\theta)^\top\delta
+
\frac{1}{2}\delta^\top H\delta
$$

Eigenvalues of the Hessian $H$ describe curvature directions. Large positive eigenvalues indicate sharp directions; negative eigenvalues indicate local descent directions away from a saddle or maximum.

## Common Equation

Eigenvalues are roots of the characteristic equation:

$$
\det(A-\lambda I)=0
$$

where $I$ is the identity matrix.

## Checks

- Is the matrix square?
- Is it symmetric? Symmetric matrices have real eigenvalues and orthogonal eigenvectors.
- Are eigenvalues being used for variance, stability, graph spectra, or curvature?
- Are repeated or near-zero eigenvalues important for the interpretation?
- Is the matrix diagonalizable, symmetric, positive semidefinite, or arbitrary?
- Do large eigenvalues imply explained variance, unstable dynamics, or sharp curvature in this context?
- Are eigenvectors meaningful, or only the subspace spanned by them?
- Is the eigenbasis stable under small perturbations, especially near repeated eigenvalues?
- Is the analysis about eigenvalues of $A$, singular values of $A$, or eigenvalues of $A^\top A$?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/machine-learning/loss-landscape|Loss landscape]]
- [[concepts/architectures/state-space-model|State-space model]]
