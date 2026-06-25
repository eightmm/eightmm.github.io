---
title: Adam
tags:
  - machine-learning
  - optimization
---

# Adam

Adam is an adaptive first-order optimizer. It keeps exponential moving averages of gradients and squared gradients, then scales each parameter update by an estimate of gradient magnitude.

For gradient $g_t=\nabla_\theta \mathcal{L}_t(\theta_t)$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t \odot g_t
$$

Bias correction is:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

The parameter update is:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

where $\eta$ is the learning rate, $\beta_1$ controls momentum, $\beta_2$ controls the squared-gradient average, $\epsilon$ prevents division by zero, and $\odot$ is elementwise multiplication.

## Why It Matters

- Adam adapts update sizes per parameter.
- It often trains deep networks faster than plain stochastic gradient descent.
- It introduces optimizer state: $m_t$ and $v_t$ must be checkpointed for faithful resume.
- Learning rate, warmup, and weight decay still matter.

## Checks

- Are $\beta_1$, $\beta_2$, $\epsilon$, and $\eta$ reported?
- Is the implementation using Adam, AdamW, or a framework variant?
- Is optimizer state saved with model checkpoints?
- Are comparisons fair when optimizer and schedule differ?
- Is gradient clipping applied before or after the optimizer sees $g_t$?

## Related

- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[infra/hpc/checkpointing|Checkpointing]]
