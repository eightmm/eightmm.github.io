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

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/machine-learning/training-loop|Training loop]]
