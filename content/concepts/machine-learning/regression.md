---
title: Regression
tags:
  - machine-learning
  - supervised-learning
---

# Regression

Regression predicts a continuous target. It is common for activity values, binding affinity, energies, coordinates, expression levels, and other scalar or vector quantities.

A standard [[concepts/machine-learning/mean-squared-error|squared-error objective]] is:

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

Define the residual:

$$
e_i
=
\hat{y}_i-y_i
=
f_\theta(x_i)-y_i
$$

MSE weights large residuals quadratically:

$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{n}\sum_i e_i^2
$$

MAE weights residuals linearly:

$$
\mathcal{L}_{\mathrm{MAE}}
=
\frac{1}{n}\sum_i |e_i|
$$

So the loss choice encodes whether large errors should dominate the update.

If uncertainty is modeled with a Gaussian likelihood:

$$
p(y\mid x)
= \mathcal{N}(y;\mu_\theta(x), \sigma_\theta^2(x))
$$

then the model predicts both mean and uncertainty.

The [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] is:

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

If variance is fixed, Gaussian NLL reduces to MSE up to constants and scale:

$$
-\log p(y\mid x)
=
\frac{1}{2\sigma^2}(y-\mu_\theta(x))^2 + C
$$

A Laplace likelihood gives an absolute-error shape:

$$
p(y\mid x)
=
\frac{1}{2b}
\exp\left(
-\frac{|y-\mu_\theta(x)|}{b}
\right)
$$

so:

$$
-\log p(y\mid x)
=
\frac{|y-\mu_\theta(x)|}{b}+C
$$

## Target Semantics

Regression is not only "predict a number." The unit and transformation define the task:

- Raw value: $y$ is used directly.
- Log-transformed value: $y'=\log(y+c)$ changes error interpretation.
- Standardized value: $y'=(y-\mu)/\sigma$ requires recording train-set statistics.
- Censored value: the true value is only known to be above or below a threshold.

If the target is standardized with train-set statistics:

$$
y'
=
\frac{y-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
$$

then predictions must be transformed back before reporting metrics in physical units:

$$
\hat{y}
=
\sigma_{\mathrm{train}}\hat{y}' + \mu_{\mathrm{train}}
$$

Do not fit normalization statistics on validation or test labels.

## Output Shape

| Output | Example | Metric Caution |
| --- | --- | --- |
| scalar | affinity, energy, activity value | unit and transform dominate interpretation |
| vector | multi-property prediction | averaging can hide one bad endpoint |
| coordinate | position, pose, structure | alignment, symmetry, and atom mapping matter |
| distribution | mean and variance | calibration must be checked |
| censored value | measurement above/below threshold | ordinary MSE may be invalid |

## Evaluation Boundary

Regression metrics answer different questions:

| Metric Type | Question |
| --- | --- |
| MAE | typical absolute error in target units |
| RMSE | large-error-sensitive fit quality |
| $R^2$ | variance explained relative to a baseline |
| Pearson/Spearman | linear or rank association |
| NLL / calibration | are predicted uncertainties meaningful? |

For downstream screening, a regression model may be used as a ranker. In that case [[concepts/machine-learning/ranking|Ranking]] metrics may be more relevant than absolute error.

## Checks

- Is the target scale linear, log-transformed, standardized, or censored?
- Are outliers measurement errors or meaningful rare cases?
- Does the metric reflect downstream tolerance, such as absolute error or rank quality?
- Are uncertainty and calibration required for decision-making?
- Are train-set normalization statistics reused consistently at validation and test time?
- Are predictions reported in normalized units or original units?
- Is the model evaluated as regression, ranking, or calibrated uncertainty?
- Does the split prevent near-duplicate targets, proteins, molecules, or assays from crossing train and test?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/mean-squared-error|Mean squared error]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[entities/assay|Assay]]
