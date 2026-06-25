---
title: Conformal Prediction
tags:
  - evaluation
  - uncertainty
  - calibration
---

# Conformal Prediction

Conformal prediction wraps a model with a calibration procedure that outputs prediction sets or intervals with a target coverage level. It is useful when the system needs a coverage guarantee rather than only a confidence score.

For a target miscoverage rate $\alpha$, the goal is:

$$
\mathbb{P}
\left(
Y \in C_\alpha(X)
\right)
\ge
1-\alpha
$$

where $C_\alpha(X)$ is a prediction set or interval.

## Split Conformal Setup

Use a held-out calibration set:

$$
\mathcal{D}_{\mathrm{cal}}
=
\{(x_i,y_i)\}_{i=1}^{n}
$$

Define a nonconformity score $s(x,y)$ where larger means less compatible. For classification, one simple score is:

$$
s(x,y)
=
1 - p_\theta(y\mid x)
$$

Compute calibration scores:

$$
s_i
=
s(x_i,y_i)
$$

Choose a quantile threshold $\hat{q}_{1-\alpha}$ from the calibration scores. The prediction set is:

$$
C_\alpha(x)
=
\{y : s(x,y) \le \hat{q}_{1-\alpha}\}
$$

For regression, the nonconformity score can be absolute residual:

$$
s(x,y)
=
|y-\hat{\mu}_\theta(x)|
$$

and the interval becomes:

$$
C_\alpha(x)
=
[
\hat{\mu}_\theta(x)-\hat{q}_{1-\alpha},
\hat{\mu}_\theta(x)+\hat{q}_{1-\alpha}
]
$$

## What It Guarantees

Under exchangeability assumptions, conformal prediction gives marginal coverage:

$$
\mathbb{P}(Y \in C_\alpha(X)) \ge 1-\alpha
$$

This does not guarantee conditional coverage for every subgroup, scaffold, protein family, domain slice, or OOD region.

## Evaluation

Coverage alone is not enough. A trivial huge set can cover everything. Report:

- empirical coverage
- average set size or interval width
- coverage by important slice
- failure cases where the true label is excluded
- behavior under distribution shift

## Checks

- Is the calibration set separate from train, validation, and final test use?
- Does the exchangeability assumption match the split and deployment claim?
- Is coverage reported with set size or interval width?
- Are important slices checked for undercoverage?
- Is conformal prediction being used for uncertainty communication, abstention, or decision support?

## Related

- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
