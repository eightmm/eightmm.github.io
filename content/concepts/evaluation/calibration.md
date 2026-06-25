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

Calibration complements [[concepts/evaluation/probability-metrics|probability metrics]] such as NLL and Brier score. NLL and Brier score summarize probability quality as scalar metrics; calibration diagnostics show where confidence and empirical frequency disagree.

## Practical Checks

- Plot a reliability diagram; bin predictions and compare to empirical accuracy.
- Report expected calibration error (ECE) alongside accuracy.
- Calibrate on held-out data (temperature scaling, isotonic) — never on the test set.
- Re-check calibration under distribution shift; it degrades faster than accuracy.
- For ranking tasks, separate calibration from discrimination (AUC).
- For abstention or human review, evaluate [[concepts/evaluation/selective-prediction|selective prediction]] instead of reporting confidence alone.

## Temperature Scaling

For classification logits $z$, temperature scaling uses:

$$
p_T(y=k\mid x)
=
\frac{\exp(z_k/T)}
\sum_{\ell=1}^{K}\exp(z_\ell/T)}
$$

where $T>0$ is fit on validation data. Larger $T$ softens overconfident probabilities; smaller $T$ sharpens them.

## What Calibration Is Not

- High accuracy does not imply calibration.
- High AUC does not imply calibrated probabilities.
- A model can be calibrated on IID data and miscalibrated under OOD shift.
- Calibrating on the test set leaks evaluation information.

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/index|Evaluation]]
