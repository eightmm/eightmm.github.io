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

## Loss Decomposition

For paper notes, write the training objective as:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q_{\mathrm{train}}}
\left[
w(u)\,\ell(f_\theta(r(u)), y(u))
\right]
+ \lambda \Omega(\theta)
$$

where $u$ is the sampled training unit, $r(u)$ is the representation, $y(u)$ is the target, $w(u)$ is an optional weight, and $\Omega$ is regularization. This makes hidden assumptions visible: sampling distribution, label semantics, representation, weighting, and regularization.

## Choosing A Loss

| Target / Claim | Common Loss | Check |
| --- | --- | --- |
| calibrated class probability | cross-entropy, NLL | evaluate NLL, Brier score, calibration, not only accuracy |
| hard classification | cross-entropy, focal loss, hinge loss | threshold and class prevalence must match deployment |
| numeric regression | MSE, MAE, Huber, Gaussian NLL | unit, scale, censoring, and outlier policy define meaning |
| ranking or retrieval | pairwise, listwise, contrastive | sampled negatives must match candidate corpus |
| sequence generation | token NLL, sequence loss, RL-style objective | teacher forcing may not match free-running generation |
| molecule or structure generation | denoising, score, velocity, validity-filtered objective | loss may not imply validity, novelty, or task utility |
| pose or coordinate prediction | coordinate loss, distance loss, equivariant loss | atom mapping, symmetry, and frame must be explicit |

## Reduction Boundary

For variable-length examples, the denominator matters:

$$
\mathcal{L}
=
\frac{\sum_i \sum_{j\in \mathcal{V}_i} \ell_{ij}}
{\sum_i |\mathcal{V}_i|}
$$

where $\mathcal{V}_i$ is the valid token, residue, atom, edge, pixel, or candidate set for example $i$. Averaging per token and averaging per example can optimize different populations.

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
- Is the sampled unit an example, token, residue, atom, edge, pair, pose, or trajectory?
- Does the loss denominator match the population implied by the reported metric?
- Is the loss compatible with censored, missing, weak, or noisy labels?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/learning/supervised-learning|Supervised learning]]
