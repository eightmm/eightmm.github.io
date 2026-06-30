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

## Kernel Matrix

For training examples $x_1,\ldots,x_n$, the kernel matrix is:

$$
K_{ij}=K(x_i,x_j)
$$

A valid kernel should produce a positive semidefinite matrix:

$$
c^\top K c \ge 0
\quad
\text{for all } c\in\mathbb{R}^n
$$

This condition means the kernel can be interpreted as an inner product in some feature space.

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

Polynomial kernels use:

$$
K(x,x')
=
(x^\top x' + c)^p
$$

Linear kernels recover ordinary linear models:

$$
K(x,x')=x^\top x'
$$

Kernel ridge regression solves:

$$
\hat{\alpha}
=
(K+\lambda I)^{-1}y
$$

where $K_{ij}=K(x_i,x_j)$ and $\lambda$ is a regularization strength.

## Length Scale and Normalization

The RBF length scale controls locality:

| $\sigma$ | Behavior |
| --- | --- |
| too small | each point only resembles itself; overfitting risk |
| too large | all points look similar; underfitting risk |
| tuned | similarity matches task-relevant variation |

Because many kernels depend on distance or dot product, input scale changes the model:

$$
K(Ax,Ax')
\neq
K(x,x')
$$

in general. Standardization or domain-specific feature scaling is often part of the kernel definition.

## Kernel vs Neural Representation

| Kernel method | Neural representation |
| --- | --- |
| similarity function is specified directly | representation is learned from data |
| strong on small/medium data with good features | strong when representation learning is needed |
| training can scale poorly with $n$ | optimization can scale with minibatches |
| easier to inspect similarity assumptions | harder but more flexible |

## Watch For

- Kernel choice encodes strong assumptions.
- Scaling to large datasets can be expensive.
- Input normalization often matters.
- Similarity must match the deployment notion of generalization.
- Positive semidefinite assumptions matter for many algorithms.
- A strong kernel baseline can reveal whether a neural model is actually learning useful representations.

## When It Helps

- Small or medium datasets where a strong similarity function is known.
- Scientific settings where engineered features or distances are meaningful.
- Baseline comparisons against neural representations.

## Related

- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
