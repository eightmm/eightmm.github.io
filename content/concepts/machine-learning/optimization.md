---
title: Optimization
tags:
  - machine-learning
---

# Optimization

Optimization is the process of updating model parameters to improve an objective. In machine learning, the optimized objective is usually a proxy for the real task.

## Core Ideas

- Loss defines the training signal.
- Gradients indicate local directions for parameter updates.
- Learning rate controls update size.
- Batches estimate the objective from subsets of data.

## Watch For

- Lower training loss does not guarantee better generalization.
- Optimization instability can look like model failure.
- Hyperparameters can dominate small experiments.

## Related

- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/evaluation/index|Evaluation]]
