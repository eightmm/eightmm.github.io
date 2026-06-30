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

## Regularization as Inductive Bias

Regularization is not only a penalty term. It encodes a preference over solutions.

| Regularizer | Preference |
| --- | --- |
| L2 penalty / weight decay | smaller parameter norms |
| L1 penalty | sparse parameters |
| dropout | robustness to missing activations |
| data augmentation | invariance to transformations |
| early stopping | simpler solution reached earlier in training |
| architecture constraint | locality, permutation symmetry, equivariance, bottleneck |

This means the regularizer should match what you believe about the task. A mismatched regularizer can improve validation loss by exploiting dataset artifacts rather than the intended structure.

## Explicit and Implicit Regularization

| Type | Example | Note |
| --- | --- | --- |
| explicit | penalty term, dropout, augmentation | visible in objective or data pipeline |
| implicit | optimizer, batch size, initialization, early stopping | affects solution even without explicit penalty |
| architectural | convolution, parameter sharing, equivariance | constrains function class |
| evaluation-driven | model selection by validation | controls complexity through selection |

For example, stochastic gradient methods can prefer some solutions over others even when the written objective is unchanged.

## Regularization Path

Varying $\lambda$ gives a regularization path:

$$
\hat{\theta}(\lambda)
=
\arg\min_\theta
\hat{R}_{\mathrm{train}}(\theta)
+\lambda\Omega(\theta)
$$

Useful plots compare training and validation performance as $\lambda$ changes. If both are poor, the issue may be underfitting or feature/objective mismatch rather than insufficient regularization.

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
- Is the regularizer explicit, implicit, architectural, or selection-based?
- Does stronger regularization improve calibration or only one target metric?

## Related

- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/overfitting-underfitting|Overfitting and underfitting]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/architectures/dropout|Dropout]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
