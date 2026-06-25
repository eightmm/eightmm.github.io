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

## Batch Reduction

Implementation details matter because the optimizer receives one scalar. For per-example losses $\ell_i$:

$$
\mathcal{L}_{\mathrm{mean}}
=
\frac{1}{|B|}
\sum_{i\in B}\ell_i
$$

$$
\mathcal{L}_{\mathrm{sum}}
=
\sum_{i\in B}\ell_i
$$

Changing `mean` to `sum` changes gradient scale:

$$
\nabla_\theta \mathcal{L}_{\mathrm{sum}}
=
|B|
\nabla_\theta \mathcal{L}_{\mathrm{mean}}
$$

so the effective learning rate changes unless the update rule compensates.

For weighted data:

$$
\mathcal{L}_{B}
=
\frac{\sum_{i\in B} w_i \ell_i}{\sum_{i\in B} w_i}
$$

where $w_i$ may encode class imbalance, sampling correction, confidence, or task weighting.

## Training Loss vs Reported Metric

The optimized loss and reported metric can differ:

$$
\theta^\star
=
\arg\min_\theta \mathcal{L}_{\mathrm{train}}(\theta)
\quad
\not\Rightarrow
\quad
\arg\max_\theta M_{\mathrm{valid}}(\theta)
$$

This is normal when the loss is a smooth surrogate for a discrete or domain-specific metric. The important question is whether lower validation loss and better task metric move together.

## Common Loss Families

- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]] for categorical labels and next-token prediction.
- [[concepts/machine-learning/mean-squared-error|Mean squared error]] for regression and reconstruction.
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]] for explicit probabilistic models.
- Pairwise or listwise losses for [[concepts/machine-learning/ranking|Ranking]].
- Contrastive losses for [[concepts/learning/contrastive-learning|Contrastive learning]].

## Checks

- Does the loss match the evaluation metric?
- Are targets continuous, categorical, ordinal, structured, or pairwise?
- Is the loss reduced by mean, sum, token count, valid element count, or task-specific weights?
- Does gradient accumulation divide the loss at the right boundary?
- Does imbalance require weighting, sampling, or calibration?
- Is label noise large enough that a robust loss matters?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/learning/supervised-learning|Supervised learning]]
