---
title: Ensemble Method
tags:
  - machine-learning
---

# Ensemble Method

An ensemble method combines multiple models to improve robustness or predictive performance.

## Common Patterns

- Bagging trains models on resampled data and averages their predictions.
- Boosting trains models sequentially to correct previous errors.
- Stacking trains a meta-model over base model predictions.

## Why It Matters

Ensembles often improve tabular prediction and reduce variance, but they can also hide failure modes if evaluation is weak.

## Related

- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/leakage|Leakage]]
