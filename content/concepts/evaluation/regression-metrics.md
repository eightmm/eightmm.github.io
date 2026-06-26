---
title: Regression Metrics
tags:
  - evaluation
  - regression
  - metrics
---

# Regression Metrics

Regression metrics evaluate continuous predictions. The right metric depends on target scale, noise, outliers, tolerance, calibration, and whether the application needs absolute accuracy, ranking, or uncertainty.

Given predictions $\hat{y}_i$ and targets $y_i$, define residuals:

$$
e_i = \hat{y}_i - y_i
$$

Many regression mistakes come from ignoring the unit of $y$. A model evaluated on $K_d$, $\log K_d$, and pIC50 is not being evaluated on the same error scale.

## Metric Families

| Family | Question | Examples |
|---|---|---|
| Absolute error | How large is the typical miss? | MAE, median absolute error |
| Squared error | Are large misses strongly penalized? | MSE, RMSE |
| Explained variation | Does the model beat the mean baseline? | $R^2$ |
| Rank agreement | Does the ordering match? | Pearson, Spearman |
| Probabilistic regression | Is predictive uncertainty useful? | NLL, CRPS, calibration |

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

Median absolute error is even more robust:

$$
\operatorname{MedAE}
=
\operatorname{median}_{i}
|e_i|
$$

It is useful when a small number of extreme labels or failures should not dominate the summary.

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

MSE and RMSE are appropriate when large errors are disproportionately costly. They are fragile when the label distribution contains experimental artifacts, mixed protocols, or long-tailed noise.
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

$R^2$ can be negative on held-out data. This means the model is worse than predicting the test-set mean:

$$
\sum_i (y_i-\hat{y}_i)^2
>
\sum_i (y_i-\bar{y})^2
$$

Do not compare $R^2$ across datasets with very different target variance without checking the denominator.

## Correlation

Pearson correlation measures linear association:

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

Spearman correlation measures rank association:

$$
\rho
=
\operatorname{corr}
\left(
\operatorname{rank}(\hat{y}),
\operatorname{rank}(y)
\right)
$$

Correlation can be high even when predictions are badly calibrated in absolute units. This is common when a model ranks compounds correctly but compresses or shifts the predicted values.

## Log Scales and Bioactivity

Binding and activity labels are often logarithmic:

$$
\mathrm{pIC}_{50}
=
-\log_{10}
\left(
\frac{\mathrm{IC}_{50}}{1\,\mathrm{M}}
\right)
$$

An error of $1.0$ in pIC50 corresponds to a tenfold error in concentration scale. This makes p-scale errors easier to compare across orders of magnitude, but it also means the metric is not the same as absolute concentration error.

Use a table like this before comparing papers:

| Target form | Example | Metric caution |
|---|---|---|
| Raw concentration | IC50, $K_d$ | scale is long-tailed |
| Log concentration | pIC50, p$K_d$ | errors are fold-change-like |
| Standardized label | z-score | unit is dataset-specific |
| Censored label | $>$ or $<$ threshold | ordinary MAE/RMSE can be biased |

See [[concepts/data/censored-label|Censored label]] for thresholded measurements.

## Probabilistic Regression

If the model predicts a distribution, not only a point estimate, evaluate the distribution:

$$
p_\theta(y\mid x)
$$

For Gaussian regression:

$$
p_\theta(y_i\mid x_i)
=
\mathcal{N}
\left(
y_i;\mu_\theta(x_i),\sigma_\theta^2(x_i)
\right)
$$

The negative log-likelihood is:

$$
\operatorname{NLL}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\frac{1}{2}\log \sigma_i^2
+
\frac{(y_i-\mu_i)^2}{2\sigma_i^2}
\right]
+ C
$$

This metric rewards accurate means and honest uncertainty. It can be gamed by inflating uncertainty, so pair it with calibration diagnostics.

## Choosing the Metric

| Claim | Stronger metric choice |
|---|---|
| "The model predicts values accurately." | MAE/RMSE with units |
| "The model prioritizes candidates." | Spearman, enrichment, top-k error |
| "The model improves over a baseline." | paired comparison and confidence interval |
| "The model knows when it is uncertain." | probabilistic NLL and calibration |
| "The model works across scaffolds/families." | split-specific metrics and error analysis |

## Checks

- Is the target linear, logarithmic, standardized, censored, or clipped?
- Are outliers meaningful or measurement artifacts?
- Does the application care about absolute error, squared error, rank order, or threshold crossing?
- Are units and transformations reported clearly?
- Are confidence intervals or repeated splits needed for metric stability?
- Is the train/test split compatible with the claim?
- Are uncertainty estimates evaluated separately from point prediction?
- Are errors stratified by scaffold, protein family, label source, or target range?

## Related

- [[concepts/machine-learning/regression|Regression]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
