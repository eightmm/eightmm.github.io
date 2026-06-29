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

## Selection Effects

An estimator can become optimistic when it is repeatedly used for model selection:

$$
\hat{Q}_{\mathrm{selected}}
=
\max_{k\in\{1,\ldots,K\}}\hat{Q}_k
$$

Even if each $\hat{Q}_k$ is individually reasonable, the maximum over many candidates is biased upward. This is why validation metrics and final test metrics must be separated.

| Use | Correct role |
| --- | --- |
| training loss | optimize parameters |
| validation metric | choose model, threshold, prompt, checkpoint, or hyperparameter |
| final test metric | estimate performance once after selection is fixed |
| deployment metric | monitor behavior under target operating conditions |

## Metric Object

Before interpreting a score, identify whether the metric is:

| Metric type | Example | Extra question |
| --- | --- | --- |
| per-example average | MSE, accuracy, NLL | are examples independent and representative? |
| ranking statistic | AUROC, enrichment, NDCG | what is the query/candidate denominator? |
| thresholded decision | F1, precision, recall | how was threshold selected? |
| generated-sample statistic | validity, novelty, diversity | attempted or post-filtered samples? |
| calibrated probability score | Brier score, NLL, ECE | are probabilities meaningful for decisions? |

## Checks

- What distribution does the expectation refer to?
- Is the reported metric an unbiased or biased estimate of the desired quantity?
- Was the same metric used for training, validation, and final reporting?
- Is uncertainty reported with confidence intervals, bootstrap, paired tests, or seed variance?
- Does the metric support the paper's claim, or only a narrower benchmark statement?
- Was the metric used many times during model, prompt, threshold, or checkpoint selection?
- Does the metric denominator match the objects claimed in the text?

## Related

- [[math/evaluation-math|Evaluation Math]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
