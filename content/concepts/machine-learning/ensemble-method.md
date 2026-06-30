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

## Bias and Variance View

For independent base predictors with variance $\sigma^2$, averaging reduces variance:

$$
\operatorname{Var}
\left[
\frac{1}{M}\sum_{m=1}^{M} f_m(x)
\right]
=
\frac{\sigma^2}{M}
$$

If predictors are correlated with pairwise correlation $\rho$, the reduction is weaker:

$$
\operatorname{Var}(f_{\mathrm{ens}})
\approx
\sigma^2
\left(
\rho + \frac{1-\rho}{M}
\right)
$$

This is why model diversity matters. Many identical models do not make a strong ensemble.

## Ensemble Types

| Type | Main idea | Typical use |
| --- | --- | --- |
| Bagging | average models trained on resampled data | reduce variance of unstable learners |
| Random forest | bagging plus random feature splits | robust tabular baseline |
| Boosting | sequentially add weak learners | strong tabular prediction |
| Stacking | train a meta-model over base predictions | combine heterogeneous models |
| Snapshot/seed ensemble | average neural checkpoints or seeds | reduce stochastic training variance |

## Evaluation Boundary

Ensembling must be evaluated on data not used to choose base models, weights, or stacking meta-features.

| Risk | Why |
| --- | --- |
| stacking on validation predictions then reporting same validation score | meta-model leaks validation labels |
| selecting ensemble members by test performance | test set becomes training signal |
| averaging uncalibrated probabilities | accuracy may improve while calibration worsens |
| ensemble hides subgroup failure | average metric improves but slices degrade |

For classification, calibration should be checked after ensembling, not assumed from better accuracy.

## Checks

- Were base models trained without leaking validation or test labels?
- Does ensembling improve the target metric or only average away calibration errors?
- Is the compute cost acceptable for the deployment path?
- Does model diversity come from data, features, architecture, seed, or objective?
- Were ensemble weights selected on validation data only?
- Does performance improve on relevant slices, not just the average metric?

## Related

- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/leakage|Leakage]]
