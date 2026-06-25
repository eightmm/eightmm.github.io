---
title: Regularization
tags:
  - machine-learning
---

# Regularization

Regularization constrains a model or training process to reduce overfitting and improve generalization.

A common regularized objective is:

$$
\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
+ \lambda \Omega(\theta)
$$

Here $\Omega(\theta)$ is a penalty term and $\lambda$ controls its strength.

## Common Forms

- L1 or L2 penalties on parameters.
- Early stopping based on validation performance.
- Data augmentation.
- Dropout or noise injection.
- Architectural constraints such as locality or equivariance.

## Why It Matters

More capacity is not automatically better. Regularization makes the learned function less dependent on accidental patterns in the training data.

## Related

- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
