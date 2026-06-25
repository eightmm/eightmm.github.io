---
title: Regression
tags:
  - machine-learning
  - supervised-learning
---

# Regression

Regression predicts a continuous target. It is common for activity values, binding affinity, energies, coordinates, expression levels, and other scalar or vector quantities.

A standard squared-error objective is:

$$
\mathcal{L}_{\mathrm{MSE}}
= \frac{1}{n}\sum_{i=1}^{n}
\lVert f_\theta(x_i)-y_i\rVert_2^2
$$

Mean absolute error is less sensitive to large residuals:

$$
\mathcal{L}_{\mathrm{MAE}}
= \frac{1}{n}\sum_{i=1}^{n}
\lvert f_\theta(x_i)-y_i\rvert
$$

If uncertainty is modeled with a Gaussian likelihood:

$$
p(y\mid x)
= \mathcal{N}(y;\mu_\theta(x), \sigma_\theta^2(x))
$$

then the model predicts both mean and uncertainty.

## Checks

- Is the target scale linear, log-transformed, standardized, or censored?
- Are outliers measurement errors or meaningful rare cases?
- Does the metric reflect downstream tolerance, such as absolute error or rank quality?
- Are uncertainty and calibration required for decision-making?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/calibration|Calibration]]
- [[entities/assay|Assay]]
