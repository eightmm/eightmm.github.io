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

## Why It Matters

- Training objectives average losses over data.
- Evaluation estimates risk over a held-out distribution.
- Sampling changes which expectation the model optimizes.
- Monte Carlo estimates approximate intractable integrals.

## Checks

- Which distribution is the expectation under?
- Is the estimate biased by sampling, filtering, or missing data?
- Is the average over examples, tokens, classes, tasks, seeds, or folds?
- Does the reported metric include uncertainty or variance?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/machine-learning/training-loop|Training loop]]
