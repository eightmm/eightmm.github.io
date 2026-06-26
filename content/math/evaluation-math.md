---
title: Evaluation Math
tags:
  - math
  - evaluation
---

# Evaluation Math

Evaluation math separates model performance claims from noise, leakage, calibration problems, and benchmark artifacts.

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{test}}}
[\mathcal{L}(f(x), y)]
$$

The empirical test score is only an estimate of this target.

## Core Notes

- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]

## Estimate and Uncertainty

For a test set of size $m$, the empirical risk is:

$$
\hat{R}
=
\frac{1}{m}\sum_{j=1}^{m}
\ell_j
$$

where $\ell_j=\mathcal{L}(f(x_j),y_j)$ is the per-example loss.

An approximate standard error is:

$$
\operatorname{SE}(\hat{R})
\approx
\frac{s_\ell}{\sqrt{m}}
$$

where $s_\ell$ is the sample standard deviation of per-example losses. This approximation is weak when examples are dependent, heavily stratified, or selected after model tuning.

## Comparison Map

| Comparison | Preferred Evidence | Risk |
| --- | --- | --- |
| Same examples, two models | paired difference and confidence interval | aggregate scores hide per-example dependence |
| Many seeds | mean, variance, and selection rule | best seed is mistaken for expected performance |
| Many prompts/checkpoints | held-out selection protocol | repeated trials inflate false discoveries |
| Many datasets | per-dataset effect size | average score hides failures on important strata |
| Imbalanced classification | PR-AUC, enrichment, calibrated threshold | ROC-AUC can look good under severe imbalance |

For paired comparison:

$$
\Delta
=
\frac{1}{m}\sum_{j=1}^{m}
(s_{A,j}-s_{B,j})
$$

where $s_{A,j}$ and $s_{B,j}$ are scores or losses for two systems on the same example.

## Probability Quality

- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/conformal-prediction|Conformal prediction]]

## Checks

- Is the reported score a point estimate or an uncertainty-aware comparison?
- Is the test set independent from model selection?
- Are comparisons paired on the same examples?
- Is the metric aligned with the real decision?
- Are confidence, calibration, and abstention needed?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[bio/data-evaluation|Bio data and evaluation]]
