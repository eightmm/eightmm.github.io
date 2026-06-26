---
title: Calculus and Optimization
tags:
  - math
  - optimization
---

# Calculus and Optimization

Calculus explains how model parameters change during training. Optimization turns gradients into updates.

$$
\theta_{t+1}
=
\theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

## Core Notes

- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]

## Training Stability

- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/loss-landscape|Loss landscape]]

## Checks

- Is the gradient taken with respect to parameters, inputs, coordinates, or time?
- Is the loss differentiable where the model is trained?
- Is instability caused by learning rate, initialization, normalization, batch size, or data scale?
- Are micro-steps, optimizer steps, and consumed samples counted separately?

## Related

- [[math/index|Math]]
- [[ai/machine-learning|Machine Learning]]
- [[infra/training/index|Training infra]]
