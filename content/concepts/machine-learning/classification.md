---
title: Classification
tags:
  - machine-learning
  - supervised-learning
---

# Classification

Classification predicts a discrete class label. The model usually outputs a probability distribution over classes, then a decision rule converts that distribution into a class.

For $K$ classes, logits $z\in\mathbb{R}^{K}$ become probabilities through softmax:

$$
p_\theta(y=k\mid x)
= \frac{\exp(z_k)}
{\sum_{\ell=1}^{K}\exp(z_\ell)}
$$

The predicted class is:

$$
\hat{y}
= \arg\max_k p_\theta(y=k\mid x)
$$

The standard supervised loss is [[concepts/machine-learning/cross-entropy-loss|cross-entropy]]:

$$
\mathcal{L}_{\mathrm{CE}}
= -\sum_{k=1}^{K} y_k\log p_\theta(y=k\mid x)
$$

For binary classification, this becomes:

$$
\mathcal{L}_{\mathrm{BCE}}
=
-y\log p_\theta(y=1\mid x)
-(1-y)\log(1-p_\theta(y=1\mid x))
$$

## Decision Rule

Probabilities and decisions are different. A threshold $\tau$ converts a probability into a label:

$$
\hat{y}
=
\mathbb{1}\left[p_\theta(y=1\mid x)\ge \tau\right]
$$

Changing $\tau$ changes precision, recall, false positives, and false negatives without changing the model itself.

## Multi-Label Case

If labels are not mutually exclusive, each class often uses an independent sigmoid:

$$
p_k = \sigma(z_k),
\qquad
\hat{y}_k=\mathbb{1}[p_k\ge \tau_k]
$$

This is different from softmax classification, where probabilities compete across classes.

## Checks

- Are labels mutually exclusive, multi-label, ordinal, or hierarchical?
- Is class imbalance severe enough to require weighting or threshold tuning?
- Are probabilities calibrated, or only rankings/classes needed?
- Does the split prevent near-duplicate examples from crossing train and test?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/architectures/softmax|Softmax]]
