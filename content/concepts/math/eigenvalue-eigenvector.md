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

## Why It Matters

- PCA uses eigenvectors of a covariance matrix.
- Optimization stability is shaped by curvature and [[concepts/math/jacobian-hessian|Hessian]] eigenvalues.
- Graph methods use eigenvectors of adjacency or Laplacian matrices.
- Dynamical systems and state-space models often analyze eigenvalues for stability.

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

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/architectures/state-space-model|State-space model]]
