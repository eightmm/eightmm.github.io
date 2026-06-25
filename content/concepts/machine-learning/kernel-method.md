---
title: Kernel Method
tags:
  - machine-learning
---

# Kernel Method

A kernel method uses a similarity function to model nonlinear relationships without explicitly constructing all transformed features.

The kernel is an inner product in an implicit feature space:

$$
K(x, x') = \langle \phi(x), \phi(x') \rangle
$$

A common prediction form is:

$$
f(x) = \sum_{i=1}^{n} \alpha_i K(x_i, x) + b
$$

## Intuition

The kernel defines what it means for two examples to be similar. A model can then behave like a linear method in an implicit high-dimensional feature space.

## Examples

- Support vector machine with an RBF kernel.
- Kernel ridge regression.
- Gaussian process models.

The radial basis function kernel is:

$$
K_{\mathrm{RBF}}(x,x')
=
\exp\left(
-\frac{\lVert x-x'\rVert_2^2}{2\sigma^2}
\right)
$$

where $\sigma$ controls the similarity length scale.

Kernel ridge regression solves:

$$
\hat{\alpha}
=
(K+\lambda I)^{-1}y
$$

where $K_{ij}=K(x_i,x_j)$ and $\lambda$ is a regularization strength.

## Watch For

- Kernel choice encodes strong assumptions.
- Scaling to large datasets can be expensive.
- Input normalization often matters.
- Similarity must match the deployment notion of generalization.

## When It Helps

- Small or medium datasets where a strong similarity function is known.
- Scientific settings where engineered features or distances are meaningful.
- Baseline comparisons against neural representations.

## Related

- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
