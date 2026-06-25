---
title: Gradient Clipping
tags:
  - machine-learning
  - optimization
  - training
---

# Gradient Clipping

Gradient clipping limits gradient size before an optimizer update. It is used to reduce instability from exploding gradients, especially in sequence models, reinforcement learning, and very deep networks.

For a gradient vector $g_t$, norm clipping with threshold $c$ is:

$$
\tilde{g}_t
=
g_t
\cdot
\min\left(
1,
\frac{c}{\lVert g_t \rVert_2}
\right)
$$

If $\lVert g_t\rVert_2 \le c$, the gradient is unchanged. If it is larger, the gradient is rescaled to have norm $c$.

## Where It Fits

A common training step is:

$$
g_t = \nabla_\theta \mathcal{L}_t(\theta_t)
\quad
\rightarrow
\quad
\tilde{g}_t = \operatorname{clip}(g_t)
\quad
\rightarrow
\quad
\theta_{t+1} = \operatorname{optimizer}(\theta_t, \tilde{g}_t)
$$

Gradient clipping changes optimization dynamics. It should be recorded as part of the training configuration.

## Checks

- Is clipping by global norm, per-parameter norm, or value?
- Is clipping applied before or after gradient scaling and accumulation?
- Are distributed gradients clipped consistently across workers?
- Is clipping hiding a deeper instability such as bad learning rate or loss scaling?
- Are gradient norms logged enough to know whether clipping is active?

## Related

- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/systems/training-run|Training run]]
