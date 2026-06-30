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

## What Adam Is Estimating

Adam stores two optimizer-state tensors for each optimized parameter:

| State | Estimate | Effect |
| --- | --- | --- |
| $m_t$ | first moment of gradient | smooth update direction |
| $v_t$ | second raw moment of gradient | scale update by recent gradient magnitude |

The update can be read per coordinate:

$$
\Delta \theta_{t,i}
=
-\eta
\frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}}+\epsilon}
$$

Coordinates with consistently large squared gradients get smaller effective steps. Coordinates with small squared gradients get larger effective steps. This is why Adam is often robust early in training, but it also means that optimizer choice, learning-rate schedule, and parameter grouping are part of the model claim.

## Bias Correction

At the start of training, $m_t$ and $v_t$ are biased toward zero because they are initialized at zero. Bias correction compensates for this:

$$
\mathbb{E}[m_t]
\approx
(1-\beta_1^t)\mathbb{E}[g_t]
$$

so dividing by $1-\beta_1^t$ makes the early estimate comparable in scale. Without bias correction, the first updates can be too small or behave differently across implementations.

## Epsilon and Numerical Boundary

The $\epsilon$ term is not just cosmetic:

$$
\sqrt{\hat{v}_{t,i}}+\epsilon
$$

sets a lower bound on the denominator. If $\hat{v}_{t,i}$ is tiny, $\epsilon$ controls the effective step size. Different frameworks or fused kernels may place $\epsilon$ inside or outside the square root, which can change behavior in low-precision or small-gradient regimes.

## Adam vs AdamW

Adam with an $L_2$ penalty and [[concepts/machine-learning/adamw|AdamW]] are not the same for adaptive updates.

| Choice | Update Meaning |
| --- | --- |
| Adam + $L_2$ penalty | decay term is mixed into $g_t$ and rescaled by adaptive moments |
| AdamW | decay term is applied directly to parameters |

Use Adam when describing the adaptive moment mechanism. Use AdamW when the training recipe depends on decoupled weight decay.

## Checkpoint Boundary

A faithful resume needs:

$$
(\theta_t, m_t, v_t, t, \eta_t, \text{scheduler state})
$$

Saving only $\theta_t$ changes the next update because bias correction, scheduler phase, and moment estimates are lost. This matters for long training, interrupted HPC jobs, and paper reproduction.

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
- Is $\epsilon$ placement or fused optimizer behavior relevant?
- Are bias correction, scheduler state, and step count restored on resume?
- Are parameter groups using the same learning rate and decay settings?

## Related

- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[infra/hpc/checkpointing|Checkpointing]]
