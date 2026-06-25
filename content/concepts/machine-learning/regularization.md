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

L2 regularization uses:

$$
\Omega(\theta)=\lVert \theta\rVert_2^2
$$

L1 regularization uses:

$$
\Omega(\theta)=\lVert \theta\rVert_1
=\sum_j |\theta_j|
$$

In gradient descent, an L2 penalty adds a shrinkage term:

$$
\theta_{t+1}
=
\theta_t
-\eta
\left(
\nabla_\theta \mathcal{L}_{\mathrm{train}}(\theta_t)
+2\lambda\theta_t
\right)
$$

where $\eta$ is the learning rate.

## Why It Matters

More capacity is not automatically better. Regularization makes the learned function less dependent on accidental patterns in the training data.

## Model Selection

Regularization strength is a hyperparameter. It should be selected on validation data:

$$
\lambda^\*
=
\arg\min_{\lambda}
\hat{R}_{\mathrm{val}}(f_{\hat{\theta}(\lambda)})
$$

The test set should only be used after this choice is fixed.

## Checks

- Is regularization reducing validation error, not only training loss?
- Was the regularization strength selected without using test labels?
- Does the regularizer match the intended inductive bias?
- Is the method preventing overfit, or hiding leakage in the evaluation protocol?

## Related

- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/architectures/dropout|Dropout]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
