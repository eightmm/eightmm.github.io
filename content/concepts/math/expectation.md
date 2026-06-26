---
title: Expectation
tags:
  - math
  - probability
---

# Expectation

Expectation is an average under a probability distribution. Many objectives and evaluation metrics are expectations.

For a discrete random variable:

$$
\mathbb{E}_{x\sim p}[f(x)]
= \sum_x p(x)f(x)
$$

For a continuous random variable:

$$
\mathbb{E}_{x\sim p}[f(x)]
= \int p(x)f(x)\,dx
$$

Empirical averages approximate expectations from finite data:

$$
\frac{1}{n}\sum_{i=1}^{n} f(x_i)
\approx
\mathbb{E}_{x\sim p_{\mathrm{data}}}[f(x)]
$$

Expected risk is the population version of a supervised loss:

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\mathcal{L}(f(x),y)
\right]
$$

Empirical risk replaces the unknown distribution with a dataset:

$$
\hat{R}(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f(x_i),y_i)
$$

The gap between $R$ and $\hat{R}$ is why sampling, splits, confidence intervals, and distribution shift matter.

## Linearity

Expectation is linear:

$$
\mathbb{E}[aX+bY]
=
a\mathbb{E}[X]
+
b\mathbb{E}[Y]
$$

This holds even when $X$ and $Y$ are not independent. Many loss decompositions and metric averages use this property.

## Conditional Expectation

Conditional expectation averages under a conditional distribution:

$$
\mathbb{E}[Y\mid X=x]
=
\int y\,p(y\mid x)\,dy
$$

For regression with squared error, the optimal prediction is the conditional mean:

$$
f^\*(x)
=
\mathbb{E}[Y\mid X=x]
$$

This explains why different losses imply different target summaries.

## Weighted Expectations

Sampling or class weighting changes the expectation:

$$
\mathbb{E}_{q}
\left[
w(X,Y)\mathcal{L}(f(X),Y)
\right]
$$

If $q$ is the training sampler and deployment follows $p$, the metric should make clear which expectation is being estimated.

## Why It Matters

- Training objectives average losses over data.
- Evaluation estimates risk over a held-out distribution.
- Sampling changes which expectation the model optimizes.
- Monte Carlo estimates approximate intractable integrals.
- Reinforcement learning, generative modeling, and evaluation all differ mainly in which distribution the expectation is taken under.

## Checks

- Which distribution is the expectation under?
- Is the estimate biased by sampling, filtering, or missing data?
- Is the average over examples, tokens, classes, tasks, seeds, or folds?
- Does the reported metric include uncertainty or variance?
- Is the empirical average weighted by the sampling process?
- Are macro, micro, per-class, per-token, and per-example averages being mixed?
- Does the loss target a mean, median, mode, quantile, ranking, or calibrated probability?
- Is the expectation over data, model samples, seeds, folds, or annotators?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/random-variable|Random variable]]
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/machine-learning/training-loop|Training loop]]
