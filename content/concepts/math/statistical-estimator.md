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

For example, the sample mean estimates a population mean $\mu=\mathbb{E}[X]$:

$$
\hat{\mu}
=
\frac{1}{n}\sum_{i=1}^{n}X_i
$$

An empirical risk estimate is also an estimator:

$$
\hat{R}(f)
=
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f(x_i),y_i)
$$

It estimates the population risk:

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\mathcal{L}(f(x),y)
\right]
$$

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

This decomposition is useful because a low-variance estimator can still be wrong if it is biased, and an unbiased estimator can still be noisy when $n$ is small.

## Standard Error

The standard error describes estimator variability across repeated samples. For the sample mean:

$$
\operatorname{SE}(\hat{\mu})
=
\frac{\sigma}{\sqrt{n}}
$$

When $\sigma$ is unknown, it is usually estimated with the sample standard deviation:

$$
\widehat{\operatorname{SE}}(\hat{\mu})
=
\frac{s}{\sqrt{n}}
$$

This is the bridge from an estimator to a [[concepts/evaluation/confidence-interval|confidence interval]].

## Consistency

An estimator is consistent if it converges to the target as sample size grows:

$$
\hat{\theta}_n
\xrightarrow[]{p}
\theta
\quad
\text{as}
\quad
n\to\infty
$$

Consistency does not guarantee that the estimate is reliable for a small benchmark or a shifted deployment distribution.

## Why It Matters

- A reported metric is an estimate, not the true performance.
- Small test sets have high estimator variance.
- Biased sampling makes even precise estimates misleading.
- Confidence intervals describe uncertainty of an estimator.
- Paired evaluation can reduce estimator variance by comparing models on the same examples.
- Leakage or duplicated samples can make the effective sample size smaller than the nominal $n$.

## Checks

- What population or deployment distribution is the estimator supposed to represent?
- Are samples independent, paired, clustered, duplicated, or time-correlated?
- Is uncertainty reported as variance, standard error, confidence interval, or bootstrap interval?
- Was the estimator used for model selection and final reporting on the same data?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
