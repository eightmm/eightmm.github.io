---
title: Estimator vs Metric
tags:
  - math
  - evaluation
---

# Estimator vs Metric

An estimator approximates an unknown quantity from finite data. A metric is a reported score used to compare models or make decisions. They often look similar, but they answer different questions.

$$
Q = \mathbb{E}_{z\sim p}[q(z)],
\qquad
\hat{Q}_n = \frac{1}{n}\sum_{i=1}^{n} q(z_i)
$$

Here $Q$ is a population quantity and $\hat{Q}_n$ is a finite-sample estimator. A metric may be $\hat{Q}_n$, a transformed score, a ranking statistic, or a thresholded decision value.

## Distinctions

| Term | Meaning | Failure mode |
| --- | --- | --- |
| population quantity | target value under a distribution | distribution not specified |
| estimator | finite-data approximation | biased, high variance, or selected repeatedly |
| training loss | optimized quantity | may not match final metric |
| validation metric | model selection signal | overfit through repeated selection |
| test metric | final reported score | weak if split does not match claim |
| decision metric | score tied to use case | may be absent from benchmark |

## Checks

- What distribution does the expectation refer to?
- Is the reported metric an unbiased or biased estimate of the desired quantity?
- Was the same metric used for training, validation, and final reporting?
- Is uncertainty reported with confidence intervals, bootstrap, paired tests, or seed variance?
- Does the metric support the paper's claim, or only a narrower benchmark statement?

## Related

- [[math/evaluation-math|Evaluation Math]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
