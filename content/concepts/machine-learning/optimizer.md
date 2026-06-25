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

With bias correction:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

and the Adam update is:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

[[concepts/machine-learning/adamw|AdamW]] separates adaptive optimization from weight decay:

$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

where $\lambda$ is the decoupled weight decay coefficient.

## Checks

- Is weight decay applied to the intended parameter groups?
- Are bias, normalization, embedding, and matrix parameters treated appropriately?
- Is the optimizer state saved in checkpoints?
- Does the optimizer choice matter after controlling learning rate and schedule?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/gpu-memory|GPU memory]]
