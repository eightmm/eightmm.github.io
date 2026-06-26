---
title: Cross-Entropy Loss
tags:
  - machine-learning
  - loss
  - classification
---

# Cross-Entropy Loss

Cross-entropy loss trains a probabilistic classifier by penalizing low probability assigned to the target class or target distribution.

For a target distribution $y\in\Delta^{K-1}$ and predicted probabilities $p_\theta(\cdot\mid x)$:

$$
\mathcal{L}_{\mathrm{CE}}(x,y)
=
-
\sum_{k=1}^{K}
y_k \log p_\theta(y=k\mid x)
$$

For a one-hot target with true class $c$:

$$
\mathcal{L}_{\mathrm{CE}}(x,c)
=
-
\log p_\theta(y=c\mid x)
$$

With logits $z\in\mathbb{R}^{K}$:

$$
p_\theta(y=k\mid x)
=
\operatorname{softmax}(z)_k
=
\frac{\exp(z_k)}
{\sum_{\ell=1}^{K}\exp(z_\ell)}
$$

## Binary Cross-Entropy

For binary classification with probability $p_\theta(y=1\mid x)$:

$$
\mathcal{L}_{\mathrm{BCE}}
=
-
y\log p_\theta(y=1\mid x)
-
(1-y)\log(1-p_\theta(y=1\mid x))
$$

Multi-label classification usually applies binary cross-entropy independently to each label, not softmax cross-entropy across mutually exclusive classes.

## Target Types

| Target Type | Loss Form | Main Check |
| --- | --- | --- |
| one-hot class | $-\log p_c$ | classes must be mutually exclusive |
| soft label | $-\sum_k y_k\log p_k$ | target distribution should sum to one |
| binary label | BCE | threshold and prevalence affect reported metrics |
| multi-label | independent BCE per label | labels are not mutually exclusive |
| label-smoothed class | cross-entropy with smoothed target | report smoothing value and calibration impact |

If labels are ordinal, censored, weak, or noisy, plain cross-entropy may be a convenient surrogate but not a complete statement of the label semantics.

## Relation to Likelihood

Cross-entropy on one-hot labels is the same training signal as [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] for categorical labels:

$$
\mathcal{L}_{\mathrm{CE}}
=
-
\log p_\theta(y\mid x)
$$

It can also be written as entropy plus KL divergence:

$$
H(y,p_\theta)
=
H(y) + D_{\mathrm{KL}}(y\|p_\theta)
$$

Since $H(y)$ is fixed with respect to $\theta$, minimizing cross-entropy minimizes the KL divergence from the target distribution to the model distribution.

## Logits and Stability

Cross-entropy is usually computed from logits directly:

$$
\mathcal{L}_{\mathrm{CE}}(z,c)
=
-
z_c
+
\log\sum_{k=1}^{K}\exp(z_k)
$$

Stable implementations use a log-sum-exp trick:

$$
\log\sum_k \exp(z_k)
=
m + \log\sum_k \exp(z_k-m),
\qquad
m=\max_k z_k
$$

This avoids overflow and avoids applying softmax twice.

## Metric Boundary

Cross-entropy trains probabilities, while many reported metrics use hard decisions or rankings:

| Reported Metric | Extra Decision |
| --- | --- |
| accuracy | argmax decision rule |
| F1 | threshold and averaging rule |
| AUROC | ranking of scores |
| AUPRC | ranking under class imbalance |
| calibration | probability reliability, not only correctness |

For imbalanced bioactivity, toxicity, retrieval, or screening tasks, a lower cross-entropy can coexist with weak top-k enrichment or poor calibration.

## Checks

- Are labels mutually exclusive, multi-label, soft labels, or ordinal labels?
- Are logits passed to a numerically stable implementation, rather than softmaxed twice?
- Is class imbalance handled by weighting, sampling, or threshold tuning?
- Is the reported metric accuracy, F1, AUROC, AUPRC, calibration, or NLL?
- Is label smoothing used, and is it reported?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/calibration|Calibration]]
