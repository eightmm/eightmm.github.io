---
title: Censored Label
tags:
  - data
  - labels
  - evaluation
  - molecular-modeling
---

# Censored Label

A censored label reports a bound rather than an exact value. It appears when an assay, instrument, threshold, or reporting policy can only say that a value is above or below a limit.

Examples:

- $y < L$: value is below a lower detection limit.
- $y > U$: value is above an upper detection or assay limit.
- $y \le L$ or $y \ge U$: thresholded measurement.

The observation is not a point label:

$$
y > U
\not\Rightarrow
y = U
$$

The observed label should be represented as:

$$
\tilde{y}
=
(b,\ d,\ L)
$$

where $b$ is the bound value, $d\in\{<,\le,>,\ge\}$ is the censoring direction, and $L$ records the assay or measurement context.

## Likelihood View

For a regression target with density $p_\theta(y\mid x)$, a right-censored observation $y>U$ contributes probability mass:

$$
P_\theta(Y>U\mid x)
=
\int_U^\infty p_\theta(y\mid x)\,dy
$$

A left-censored observation $y<L$ contributes:

$$
P_\theta(Y<L\mid x)
=
\int_{-\infty}^{L} p_\theta(y\mid x)\,dy
$$

Treating censored values as exact points changes the training signal and can bias evaluation.

## Point-Label Mistake

Replacing a censored value with its bound creates a biased pseudo-label.

| Observation | Wrong point interpretation | Safer interpretation |
| --- | --- | --- |
| $y > U$ | $y=U$ | value is somewhere above $U$ |
| $y < L$ | $y=L$ | value is somewhere below $L$ |
| inactive below threshold | exact low activity | thresholded or censored activity |
| capped score | exact maximum score | value reached or exceeded cap |

This matters for regression metrics such as RMSE or MAE, because the true error cannot be computed exactly from a bound.

## Modeling Choices

| Approach | Use when | Risk |
| --- | --- | --- |
| censored likelihood | distributional regression is available | requires modeling target distribution |
| interval loss | lower/upper bounds are known | implementation complexity |
| classification by threshold | decision boundary is the goal | loses continuous information |
| ranking constraints | relative order matters | needs enough comparable pairs |
| filter censored labels | few censored examples | may create selection bias |

For a right-censored lower bound $y>U$, a prediction $\hat{y}$ should not be penalized the same way as an exact label unless the metric explicitly supports censoring.

## Evaluation Boundary

| Metric question | Need |
| --- | --- |
| exact regression error | exact labels only or censor-aware estimator |
| threshold decision | consistent threshold and censoring direction |
| ranking | valid pair construction under bounds |
| calibration | event definition that respects censored observations |
| assay comparison | harmonized units, thresholds, and reporting policy |

## Common Places

- Bioactivity measurements reported as greater-than or less-than values.
- Detection limits in assays and instruments.
- Thresholded active/inactive labels derived from continuous values.
- Clipped scores or capped ratings.

## Checks

- Is the label exact, thresholded, clipped, censored, or rounded?
- Is the censoring direction recorded?
- Are units and transforms applied before or after censoring?
- Are censored values used in regression, classification, ranking, or filtering?
- Does the metric respect censoring, or does it treat bounds as exact targets?
- Are assay-specific censoring policies preserved in the dataset schema?
- Is the censoring direction stored separately from the numeric bound?
- Does label transformation happen before or after censoring, and is that documented?

## Related

- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[entities/assay|Assay]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/target-assay-label|Target-assay-label contract]]
