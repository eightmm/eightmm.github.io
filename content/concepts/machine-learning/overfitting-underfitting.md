---
title: Overfitting and Underfitting
tags:
  - machine-learning
  - evaluation
  - generalization
---

# Overfitting and Underfitting

Overfitting and underfitting describe different failures of the learned function. Both are about the relationship between training error, validation error, model capacity, data quality, and the evaluation split.

Let $R_{\mathrm{train}}$ and $R_{\mathrm{val}}$ be empirical risks on training and validation data:

$$
R_{\mathrm{train}}(f)
=
\frac{1}{|\mathcal{D}_{\mathrm{train}}|}
\sum_{(x,y)\in\mathcal{D}_{\mathrm{train}}}
\mathcal{L}(f(x),y)
$$

$$
R_{\mathrm{val}}(f)
=
\frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_{(x,y)\in\mathcal{D}_{\mathrm{val}}}
\mathcal{L}(f(x),y)
$$

The validation gap is:

$$
\Delta_{\mathrm{val}}
=
R_{\mathrm{val}}(f)
-
R_{\mathrm{train}}(f)
$$

This gap is a diagnostic, not a proof. A small gap can still be misleading if train and validation share leaked information.

## Overfitting

Overfitting happens when a model fits accidental patterns in the training data that do not hold on held-out data.

Typical signs:

- Training loss decreases while validation loss stops improving or increases.
- Validation metrics are unstable across random seeds or split choices.
- Performance collapses under grouped, temporal, scaffold, or family splits.
- The best checkpoint was selected after many validation looks without accounting for selection pressure.

Regularization changes the objective or training process to reduce this behavior:

$$
\min_\theta
\hat{R}_{\mathrm{train}}(f_\theta)
+
\lambda \Omega(\theta)
$$

where $\Omega(\theta)$ penalizes complexity and $\lambda$ controls the strength of the penalty.

## Underfitting

Underfitting happens when the model cannot fit even the training data well.

Typical causes:

- Model class is too limited for the task.
- Features or representations discard necessary information.
- Optimization is failing because of learning rate, initialization, gradient instability, or data pipeline issues.
- Labels are noisy, inconsistent, or semantically mismatched with the task.
- The loss does not match the metric or intended output.

Underfitting is not fixed by stronger regularization. It usually needs better features, more capacity, improved optimization, cleaner labels, or a better task formulation.

## Diagnostic Table

| Pattern | Likely issue | First checks |
|---|---|---|
| High train loss, high validation loss | Underfitting or optimization failure | loss, gradients, labels, model capacity |
| Low train loss, high validation loss | Overfitting or distribution shift | split unit, leakage, regularization, dataset shift |
| Low validation loss, poor test loss | Validation overuse or mismatched split | model selection log, test boundary, split rule |
| Good IID test, poor shifted test | Weak OOD generalization | shift definition, subgroup metrics, applicability domain |

## Checks

- Compare training, validation, and test curves before changing architecture.
- Inspect the split before interpreting a gap.
- Verify preprocessing and feature selection do not see validation or test data.
- Use [[concepts/evaluation/cross-validation|cross-validation]] or repeated splits when data is small.
- Use [[concepts/evaluation/error-analysis|error analysis]] to separate capacity failures from data failures.
- Do not use test performance to tune regularization, augmentation, checkpoints, or thresholds.

## Related

- [[concepts/machine-learning/generalization|Generalization]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/error-analysis|Error analysis]]
