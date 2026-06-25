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

## Watch For

- Kernel choice encodes strong assumptions.
- Scaling to large datasets can be expensive.
- Input normalization often matters.

## Related

- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
