---
title: Optimization
tags:
  - machine-learning
---

# Optimization

Optimization is the process of updating model parameters to improve an objective. In machine learning, the optimized objective is usually a proxy for the real task.

Gradient descent updates parameters in the direction that locally reduces the objective:

$$
\theta_{t+1}
= \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Here $\eta$ is the learning rate.

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

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/evaluation/index|Evaluation]]
