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

## Relation to Likelihood

Cross-entropy on one-hot labels is the same training signal as [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] for categorical labels:

$$
\mathcal{L}_{\mathrm{CE}}
=
-
\log p_\theta(y\mid x)
$$

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
