---
title: Loss Function
tags:
  - machine-learning
  - optimization
---

# Loss Function

A loss function converts prediction error into a scalar training signal. It defines what the optimizer tries to reduce, so it must match the task, target semantics, and evaluation metric.

Empirical risk minimization averages a loss over examples:

$$
\hat{R}(\theta)
= \frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

Mean squared error is common for regression:

$$
\mathcal{L}_{\mathrm{MSE}}(\hat{y}, y)
= \lVert \hat{y}-y\rVert_2^2
$$

See [[concepts/machine-learning/mean-squared-error|Mean squared error]] for the full regression loss and likelihood interpretation.

Cross-entropy is common for categorical prediction:

$$
\mathcal{L}_{\mathrm{CE}}(p, y)
= -\sum_{k=1}^{K} y_k \log p_k
$$

Here $p_k$ is the predicted probability for class $k$, and $y_k$ is the target distribution or one-hot label.

See [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]] and [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]] for the probabilistic view.

## Common Loss Families

- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]] for categorical labels and next-token prediction.
- [[concepts/machine-learning/mean-squared-error|Mean squared error]] for regression and reconstruction.
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]] for explicit probabilistic models.
- Pairwise or listwise losses for [[concepts/machine-learning/ranking|Ranking]].
- Contrastive losses for [[concepts/learning/contrastive-learning|Contrastive learning]].

## Checks

- Does the loss match the evaluation metric?
- Are targets continuous, categorical, ordinal, structured, or pairwise?
- Does imbalance require weighting, sampling, or calibration?
- Is label noise large enough that a robust loss matters?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/learning/supervised-learning|Supervised learning]]
