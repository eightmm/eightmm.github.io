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
