---
title: Validation Curve
tags:
  - machine-learning
  - diagnostics
  - model-selection
---

# Validation Curve

A validation curve shows how training and validation performance change as a hyperparameter changes. It helps choose model complexity, regularization strength, learning rate, augmentation strength, or other settings without using the test set.

For a hyperparameter $\lambda$, train a model:

$$
\hat{\theta}_{\lambda}
=
\operatorname{Train}(\mathcal{D}_{\mathrm{train}}, \lambda)
$$

Then evaluate:

$$
R_{\mathrm{train}}(\lambda),
\qquad
R_{\mathrm{val}}(\lambda)
$$

The selected value is usually:

$$
\lambda^\*
=
\arg\min_{\lambda \in \Lambda}
R_{\mathrm{val}}(\lambda)
$$

where $\Lambda$ is the searched hyperparameter set.

## Common Uses

- Model size: width, depth, number of layers, hidden dimension.
- Regularization: weight decay, dropout, label smoothing, augmentation strength.
- Optimization: learning rate, warmup length, schedule, batch size.
- Early stopping: maximum epoch, patience, validation frequency.
- Inference settings: threshold, temperature, decoding parameters.

## Interpretation

| Curve shape | Likely meaning |
|---|---|
| Train and validation both poor across all $\lambda$ | wrong feature, loss, data, or capacity range |
| Train improves but validation degrades | hyperparameter increases overfitting |
| Validation has a broad flat optimum | robust setting; choose simpler or cheaper option |
| Validation optimum is sharp | selection may be unstable; repeat across seeds or folds |
| Validation improves at search boundary | search space may be too narrow |

## Selection Bias

The more hyperparameter values are tried, the more the validation curve can overfit validation noise:

$$
\hat{R}_{\mathrm{val}}(\lambda)
=
R(\lambda) + \epsilon_{\lambda}
$$

Selecting the minimum over many noisy estimates can make $\lambda^\*$ look better than it really is. This is why large searches need a final held-out test set or nested validation design.

## Checks

- Is the x-axis one hyperparameter or a controlled search dimension?
- Are all points trained with the same split, budget, and preprocessing contract?
- Are failed runs shown or explained?
- Is the final selection made before test evaluation?
- Does the selected $\lambda^\*$ stay stable across seeds, folds, or nearby settings?
- Is the chosen setting also practical for compute, latency, or memory constraints?

## Related

- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
