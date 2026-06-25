---
title: Censored Label
tags:
  - data
  - labels
  - evaluation
  - bio-ai
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
