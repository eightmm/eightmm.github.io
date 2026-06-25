---
title: Ensemble Method
tags:
  - machine-learning
---

# Ensemble Method

An ensemble method combines multiple models to improve robustness or predictive performance.

The generic form is a weighted combination of base models:

$$
f_{\mathrm{ens}}(x)
= \sum_{m=1}^{M} w_m f_m(x)
$$

where $f_m$ is a base model and $w_m$ is its weight.

## Common Patterns

- Bagging trains models on resampled data and averages their predictions.
- Boosting trains models sequentially to correct previous errors.
- Stacking trains a meta-model over base model predictions.

For classification, averaged class probabilities are common:

$$
p_{\mathrm{ens}}(y=k\mid x)
=
\sum_{m=1}^{M}w_m p_m(y=k\mid x)
$$

For bagging with uniform weights:

$$
f_{\mathrm{bag}}(x)
=
\frac{1}{M}\sum_{m=1}^{M}f_m(x)
$$

Boosting builds an additive model:

$$
F_M(x)
=
\sum_{m=1}^{M}\eta_m h_m(x)
$$

where each weak learner $h_m$ is chosen to reduce the current residual or gradient signal.

## Why It Matters

Ensembles often improve tabular prediction and reduce variance, but they can also hide failure modes if evaluation is weak.

## Checks

- Were base models trained without leaking validation or test labels?
- Does ensembling improve the target metric or only average away calibration errors?
- Is the compute cost acceptable for the deployment path?
- Does model diversity come from data, features, architecture, seed, or objective?

## Related

- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/leakage|Leakage]]
