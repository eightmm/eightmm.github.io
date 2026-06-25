---
title: Optimizer
tags:
  - machine-learning
  - optimization
---

# Optimizer

An optimizer turns gradients into parameter updates. Different optimizers use momentum, adaptive moments, weight decay, or second-order information to change the update rule.

For an objective $\mathcal{L}(\theta)$, the optimizer receives a gradient estimate:

$$
g_t
=
\widehat{\nabla_\theta \mathcal{L}}(\theta_t)
$$

and produces an update:

$$
\Delta\theta_t
=
U(g_t, o_t, h_t)
$$

$$
\theta_{t+1}
=
\theta_t + \Delta\theta_t
$$

where $o_t$ is optimizer state and $h_t$ denotes hyperparameters such as learning rate, momentum, and weight decay.

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

## Update Boundary

The optimizer usually steps after gradients are accumulated, synchronized, optionally clipped, and optionally unscaled for mixed precision:

$$
g_t
=
\operatorname{Reduce}
\left(
\sum_{a=1}^{A}
\nabla_\theta
\frac{\mathcal{L}_{B_a}}{A}
\right)
$$

$$
g_t'
=
\operatorname{Clip}(g_t)
$$

$$
(\theta_{t+1}, o_{t+1})
=
\operatorname{Step}(\theta_t, o_t, g_t', \eta_t)
$$

This boundary should match [[concepts/machine-learning/training-step-accounting|training step accounting]], scheduler updates, logging, and checkpointing.

## Checks

- Is weight decay applied to the intended parameter groups?
- Are bias, normalization, embedding, and matrix parameters treated appropriately?
- Is the optimizer state saved in checkpoints?
- Is the optimizer step counted separately from micro-steps?
- Are gradient clipping and mixed-precision unscaling applied before the update?
- Does the optimizer choice matter after controlling learning rate and schedule?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/gpu-memory|GPU memory]]
