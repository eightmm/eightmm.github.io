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

The negative log likelihood is:

$$
\mathcal{L}_{\mathrm{NLL}}
=
\frac{1}{2}
\left[
\log \sigma_\theta^2(x)
+
\frac{(y-\mu_\theta(x))^2}{\sigma_\theta^2(x)}
\right]
+
C
$$

where $C$ is a constant independent of the model. This loss penalizes inaccurate means and overconfident uncertainty estimates.

## Target Semantics

Regression is not only "predict a number." The unit and transformation define the task:

- Raw value: $y$ is used directly.
- Log-transformed value: $y'=\log(y+c)$ changes error interpretation.
- Standardized value: $y'=(y-\mu)/\sigma$ requires recording train-set statistics.
- Censored value: the true value is only known to be above or below a threshold.

## Checks

- Is the target scale linear, log-transformed, standardized, or censored?
- Are outliers measurement errors or meaningful rare cases?
- Does the metric reflect downstream tolerance, such as absolute error or rank quality?
- Are uncertainty and calibration required for decision-making?
- Are train-set normalization statistics reused consistently at validation and test time?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[entities/assay|Assay]]
