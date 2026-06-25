---
title: Learning Curve
tags:
  - machine-learning
  - diagnostics
  - training
---

# Learning Curve

A learning curve shows how training and validation performance change as training progresses or as the amount of data changes. It is a diagnostic tool for separating optimization failure, underfitting, overfitting, and data limitation.

For step or epoch $t$, a common curve records:

$$
R_{\mathrm{train}}(t)
=
\frac{1}{|\mathcal{D}_{\mathrm{train}}|}
\sum_{(x,y)\in\mathcal{D}_{\mathrm{train}}}
\mathcal{L}(f_{\theta_t}(x), y)
$$

$$
R_{\mathrm{val}}(t)
=
\frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_{(x,y)\in\mathcal{D}_{\mathrm{val}}}
\mathcal{L}(f_{\theta_t}(x), y)
$$

where $\theta_t$ is the model state at time $t$.

## Interpretation

| Pattern | Likely meaning | Next check |
|---|---|---|
| Train and validation both high | underfitting, optimization failure, label mismatch | loss, gradients, features, capacity |
| Train improves, validation worsens | overfitting or validation mismatch | regularization, split, leakage |
| Both improve slowly | learning rate, batch size, data difficulty | optimization and schedule |
| Train noisy, validation unstable | small validation set, high variance, unstable training | seeds, confidence intervals, gradient norms |
| Validation jumps after resume | checkpoint or preprocessing mismatch | restored state and data contract |

The curve is evidence for a diagnosis, not the diagnosis itself.

## Data-Size Curve

A second kind of learning curve varies training-set size $n$:

$$
\hat{R}_{\mathrm{val}}(n)
=
\hat{R}_{\mathrm{val}}
\left(
f_{\operatorname{Train}(\mathcal{D}_{\mathrm{train}}^{(n)})}
\right)
$$

If validation error keeps decreasing with more data, data collection or curation may matter more than architecture changes. If it saturates early, capacity, objective, label quality, or task formulation may be the bottleneck.

## Logging Requirements

Record:

- Training loss and validation metric on the same x-axis.
- Learning rate, batch size, gradient norm, and skipped/failed steps.
- Validation frequency and exact checkpoint evaluated.
- Data split version and preprocessing version.
- Random seed or repeated-run variation when curves are noisy.

## Checks

- Is the validation curve computed in evaluation mode, not training mode?
- Are train and validation losses comparable, or do they use different augmentations?
- Are validation points frequent enough for [[concepts/machine-learning/early-stopping|early stopping]]?
- Is a curve change caused by schedule, resume, data change, or code change?
- Does lower train loss actually improve [[concepts/machine-learning/generalization|generalization]]?

## Related

- [[concepts/machine-learning/overfitting-underfitting|Overfitting and underfitting]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/validation-curve|Validation curve]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
