---
title: Supervised Learning
tags:
  - supervised-learning
  - machine-learning
---

# Supervised Learning

Supervised learning fits a model to map inputs to known target labels by minimizing a loss over labeled examples. It covers classification, regression, and structured prediction.

The empirical supervised objective is:

$$
\hat{\theta}
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

The main assumption is that labels $y_i$ are meaningful for the target task and split.

## Why It Matters

- The baseline paradigm when labels are available and trustworthy.
- Directly optimizes the quantity you care about, given a clean loss.
- Provides the downstream signal that pretraining and fine-tuning are evaluated against.

## Checks

- Are labels accurate, consistent, and free of leakage from the inputs?
- Do train and test splits respect the real generalization boundary?
- Is the loss aligned with the metric used at evaluation?

## Related

- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/learning/semi-supervised-learning|Semi-supervised learning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/active-learning|Active learning]]
