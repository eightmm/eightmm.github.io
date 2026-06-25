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

The standard supervised loss is cross-entropy:

$$
\mathcal{L}_{\mathrm{CE}}
= -\sum_{k=1}^{K} y_k\log p_\theta(y=k\mid x)
$$

## Checks

- Are labels mutually exclusive, multi-label, ordinal, or hierarchical?
- Is class imbalance severe enough to require weighting or threshold tuning?
- Are probabilities calibrated, or only rankings/classes needed?
- Does the split prevent near-duplicate examples from crossing train and test?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/architectures/softmax|Softmax]]
