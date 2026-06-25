---
title: Generalization
tags:
  - machine-learning
  - evaluation
  - generalization
---

# Generalization

Generalization is the ability of a learned model to perform well on examples that were not used to fit its parameters. It is the central claim behind most machine learning results: the model should not merely memorize the training set.

For a data distribution $p(x,y)$ and loss $\mathcal{L}$, the population risk is:

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p}
\left[
\mathcal{L}(f(x), y)
\right]
$$

Training only observes a finite dataset, so it minimizes empirical risk:

$$
\hat{R}_{\mathrm{train}}(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f(x_i), y_i)
$$

The generalization gap compares observed training performance with performance on unseen data:

$$
\mathrm{gap}
=
\hat{R}_{\mathrm{test}}(f)
-
\hat{R}_{\mathrm{train}}(f)
$$

Here $f$ is the learned predictor, $(x_i,y_i)$ are training examples, $n$ is the training-set size, and $\hat{R}_{\mathrm{test}}$ is an estimate on held-out data.

## What Must Be Specified

A generalization claim should state:

- Example unit: what counts as one prediction example.
- Split unit: what must not cross train, validation, and test.
- Training distribution and target evaluation distribution.
- Model-selection rule, including validation data and checkpoint choice.
- Primary metric and uncertainty estimate.
- Known leakage channels and dataset-shift risks.

Without these pieces, "the model generalizes" is too vague to audit.

## IID and OOD

IID generalization assumes train and test examples are drawn from the same distribution:

$$
(x_{\mathrm{train}},y_{\mathrm{train}})
\sim p,
\qquad
(x_{\mathrm{test}},y_{\mathrm{test}})
\sim p
$$

OOD generalization evaluates a changed distribution:

$$
p_{\mathrm{train}}(x,y)
\ne
p_{\mathrm{test}}(x,y)
$$

Most public benchmark results are easier to interpret when they report both in-distribution performance and the drop under [[concepts/evaluation/ood-generalization|OOD generalization]].

## Failure Modes

- Low train and low test performance: underfitting, optimization failure, weak features, or label mismatch.
- Low train error but high validation/test error: overfitting, leakage-free split difficulty, or distribution shift.
- High validation performance but weak test performance: validation overuse, test-shift mismatch, or model-selection leakage.
- High benchmark performance but weak deployment behavior: benchmark does not represent deployment.

## Checks

- Is the test set held out until the final claim?
- Does validation select hyperparameters without seeing test labels?
- Is preprocessing fit only on training data?
- Does the split unit match the intended deployment claim?
- Are repeated tuning cycles counted as part of model selection?
- Are performance estimates reported with confidence intervals or split/seed variation?

## Related

- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/machine-learning/overfitting-underfitting|Overfitting and underfitting]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
