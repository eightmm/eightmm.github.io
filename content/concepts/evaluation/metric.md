---
title: Metric
tags:
  - evaluation
  - methodology
---

# Metric

A metric is the number used to summarize model behavior under a specific evaluation protocol. A good metric matches the decision that will be made from the model, not just the training loss.

For a test set $\mathcal{D}_{\mathrm{test}}=\{(x_j,y_j)\}_{j=1}^{m}$, a generic metric is:

$$
\hat{M}(f)
= \frac{1}{m}\sum_{j=1}^{m}
m(f(x_j), y_j)
$$

where $m(\cdot)$ is the per-example score.

Classification accuracy is:

$$
\operatorname{Acc}
= \frac{1}{m}\sum_{j=1}^{m}
\mathbf{1}\left[\arg\max_k p_\theta(k\mid x_j)=y_j\right]
$$

Mean absolute error is:

$$
\operatorname{MAE}
= \frac{1}{m}\sum_{j=1}^{m}
\lvert \hat{y}_j-y_j\rvert
$$

## Checks

- Does the metric match the scientific or product decision?
- Is higher or lower better, and is the scale interpretable?
- Does class imbalance require AUROC, AUPRC, balanced accuracy, or calibration?
- Are confidence intervals or multiple seeds needed?
- Is the metric valid under the chosen split?

## Related

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/machine-learning/loss-function|Loss function]]
