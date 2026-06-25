---
title: Optimizer
tags:
  - machine-learning
  - optimization
---

# Optimizer

An optimizer turns gradients into parameter updates. Different optimizers use momentum, adaptive moments, weight decay, or second-order information to change the update rule.

Stochastic gradient descent is:

$$
\theta_{t+1}
=
\theta_t - \eta g_t
$$

Momentum keeps a velocity:

$$
v_{t+1}
=
\beta v_t + g_t
$$

$$
\theta_{t+1}
=
\theta_t - \eta v_{t+1}
$$

Adam-style optimizers track moving averages of gradients and squared gradients:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

## Checks

- Is weight decay applied to the intended parameter groups?
- Are bias, normalization, embedding, and matrix parameters treated appropriately?
- Is the optimizer state saved in checkpoints?
- Does the optimizer choice matter after controlling learning rate and schedule?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/gpu-memory|GPU memory]]
