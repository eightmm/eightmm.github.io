---
title: Calibration
tags:
  - evaluation
  - methodology
  - uncertainty
---

# Calibration

Calibration measures whether predicted probabilities match observed frequencies — when a model says 80% confident, it should be right about 80% of the time. A model can be accurate yet badly calibrated, which misleads any downstream decision that uses its confidence.

Perfect calibration means:

$$
\mathbb{P}(Y=\hat{Y} \mid \hat{P}=p) = p
$$

Expected calibration error estimates the average gap between confidence and empirical accuracy:

$$
\operatorname{ECE}
= \sum_{b=1}^{B}
\frac{|B_b|}{n}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|
$$

## Practical Checks

- Plot a reliability diagram; bin predictions and compare to empirical accuracy.
- Report expected calibration error (ECE) alongside accuracy.
- Calibrate on held-out data (temperature scaling, isotonic) — never on the test set.
- Re-check calibration under distribution shift; it degrades faster than accuracy.
- For ranking tasks, separate calibration from discrimination (AUC).

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/index|Evaluation]]
