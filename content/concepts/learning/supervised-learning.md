---
title: Supervised Learning
tags:
  - supervised-learning
  - machine-learning
---

# Supervised Learning

Supervised learning fits a model to map inputs to known target labels by minimizing a loss over labeled examples. It covers classification, regression, and structured prediction.

## Why It Matters

- The baseline paradigm when labels are available and trustworthy.
- Directly optimizes the quantity you care about, given a clean loss.
- Provides the downstream signal that pretraining and fine-tuning are evaluated against.

## Checks

- Are labels accurate, consistent, and free of leakage from the inputs?
- Do train and test splits respect the real generalization boundary?
- Is the loss aligned with the metric used at evaluation?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
