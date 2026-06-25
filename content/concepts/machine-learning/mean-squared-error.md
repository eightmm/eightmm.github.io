---
title: Mean Squared Error
tags:
  - machine-learning
  - loss
  - regression
---

# Mean Squared Error

Mean squared error penalizes the squared distance between predictions and targets. It is a standard regression loss and a common reconstruction loss.

For predictions $\hat{y}_i=f_\theta(x_i)$ and targets $y_i$:

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\lVert \hat{y}_i-y_i\rVert_2^2
$$

For scalar targets:

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

The gradient with respect to a scalar prediction is:

$$
\frac{\partial}{\partial \hat{y}_i}
(\hat{y}_i-y_i)^2
=
2(\hat{y}_i-y_i)
$$

Large residuals therefore produce larger gradients than small residuals.

## Likelihood View

If observations follow a Gaussian distribution with fixed variance:

$$
y\mid x \sim \mathcal{N}(f_\theta(x), \sigma^2)
$$

then minimizing MSE is equivalent to minimizing Gaussian [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] up to constants and scale.

## Checks

- Is the target scale raw, log-transformed, standardized, clipped, or censored?
- Are large residuals meaningful rare cases or measurement artifacts?
- Does the downstream task care about squared error, absolute error, rank order, or threshold crossing?
- Are target normalization statistics computed only from the training split?
- Is MSE used as a training loss, an evaluation metric, or both?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
