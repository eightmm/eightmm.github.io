---
title: Regression Metrics
tags:
  - evaluation
  - regression
  - metrics
---

# Regression Metrics

Regression metrics evaluate continuous predictions. The right metric depends on target scale, outliers, tolerance, calibration, and whether ranking or absolute accuracy matters.

Given predictions $\hat{y}_i$ and targets $y_i$, define residuals:

$$
e_i = \hat{y}_i - y_i
$$

## Mean Absolute Error

Mean absolute error measures average absolute deviation:

$$
\operatorname{MAE}
=
\frac{1}{n}
\sum_{i=1}^{n}
|e_i|
$$

MAE is interpretable in the same unit as the target and is less sensitive to large outliers than squared error.

## Mean Squared Error and RMSE

Mean squared error penalizes large errors strongly:

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
e_i^2
$$

Root mean squared error returns to the target unit:

$$
\operatorname{RMSE}
=
\sqrt{\operatorname{MSE}}
$$

## Coefficient of Determination

$R^2$ compares the model to predicting the target mean:

$$
R^2
=
1
-
\frac{
\sum_i (y_i-\hat{y}_i)^2
}{
\sum_i (y_i-\bar{y})^2
}
$$

where $\bar{y}$ is the mean target in the evaluated set.

## Correlation

Correlation measures whether predictions track relative variation:

$$
r
=
\frac{
\sum_i (\hat{y}_i-\bar{\hat{y}})(y_i-\bar{y})
}{
\sqrt{\sum_i(\hat{y}_i-\bar{\hat{y}})^2}
\sqrt{\sum_i(y_i-\bar{y})^2}
}
$$

Correlation can be high even when predictions are badly calibrated in absolute units.

## Checks

- Is the target linear, logarithmic, standardized, censored, or clipped?
- Are outliers meaningful or measurement artifacts?
- Does the application care about absolute error, squared error, rank order, or threshold crossing?
- Are units and transformations reported clearly?
- Are confidence intervals or repeated splits needed for metric stability?

## Related

- [[concepts/machine-learning/regression|Regression]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
