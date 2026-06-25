---
title: Statistical Estimator
tags:
  - math
  - statistics
  - evaluation
---

# Statistical Estimator

A statistical estimator uses finite data to estimate an unknown quantity. Most evaluation metrics, validation losses, calibration curves, and confidence intervals are estimators.

If $\theta$ is the true quantity and $X_1,\dots,X_n$ are samples, an estimator is:

$$
\hat{\theta}
= g(X_1,\dots,X_n)
$$

where $g$ is a function of the observed data.

## Bias

Bias measures systematic error:

$$
\operatorname{Bias}(\hat{\theta})
= \mathbb{E}[\hat{\theta}] - \theta
$$

An unbiased estimator has $\mathbb{E}[\hat{\theta}]=\theta$.

## Variance

Variance measures how much the estimator changes across samples:

$$
\operatorname{Var}(\hat{\theta})
= \mathbb{E}\left[
(\hat{\theta}-\mathbb{E}[\hat{\theta}])^2
\right]
$$

## Mean Squared Error

The mean squared error decomposes into variance and squared bias:

$$
\operatorname{MSE}(\hat{\theta})
= \mathbb{E}\left[(\hat{\theta}-\theta)^2\right]
= \operatorname{Var}(\hat{\theta})
+ \operatorname{Bias}(\hat{\theta})^2
$$

## Why It Matters

- A reported metric is an estimate, not the true performance.
- Small test sets have high estimator variance.
- Biased sampling makes even precise estimates misleading.
- Confidence intervals describe uncertainty of an estimator.

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
