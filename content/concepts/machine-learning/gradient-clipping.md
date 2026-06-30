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

## Clipping Types

| Type | Formula Shape | Use |
| --- | --- | --- |
| global norm | rescale all gradients by one factor | common default for deep networks |
| per-parameter norm | clip each tensor separately | can distort relative layer scales |
| value clipping | $\tilde{g}_{t,i}=\operatorname{clip}(g_{t,i},-c,c)$ | simple but axis-blind |
| adaptive clipping | threshold depends on parameter or layer norm | useful when scales vary strongly |

Global norm clipping uses all trainable parameters:

$$
\|g_t\|_2
=
\left(
\sum_{j=1}^{P}
\|g_{t,j}\|_2^2
\right)^{1/2}
$$

where $g_{t,j}$ is the gradient tensor for parameter group or tensor $j$.

## Order in a Training Step

A public training recipe should say where clipping happens:

$$
\text{backward}
\rightarrow
\text{accumulate}
\rightarrow
\text{synchronize}
\rightarrow
\text{unscale}
\rightarrow
\text{clip}
\rightarrow
\text{optimizer step}
$$

The exact order matters. Clipping before gradient accumulation is not the same as clipping the accumulated gradient. In distributed training, clipping local gradients before synchronization is not the same as clipping the global synchronized gradient.

## Clipping With Accumulation

If $A$ micro-batches form one optimizer step:

$$
g_t
=
\frac{1}{A}
\sum_{a=1}^{A}
g_{t,a}
$$

then the threshold $c$ should usually apply to $g_t$, not each $g_{t,a}$ independently, unless the method explicitly defines micro-batch clipping.

## Diagnostics

Track both the raw norm and clipped norm:

$$
r_t
=
\frac{\|\tilde{g}_t\|_2}{\|g_t\|_2}
$$

If $r_t \ll 1$ for many steps, clipping is a major part of the optimizer dynamics. That may be intended, but it should not be invisible in the run record.

## When Clipping Is a Symptom

| Pattern | Possible Cause |
| --- | --- |
| every step clips heavily | learning rate, loss scale, or batch construction may be wrong |
| spikes after resume | optimizer/scheduler/scaler state may not be restored |
| spikes only for some batches | data preprocessing or label outliers |
| spikes in sequence/graph tasks | long examples, bad masking, or variable-size reduction |
| NaN after clipping | numerical overflow happened before the clip point |

## Checks

- Is clipping by global norm, per-parameter norm, or value?
- Is clipping applied before or after gradient scaling and accumulation?
- Are distributed gradients clipped consistently across workers?
- Is clipping hiding a deeper instability such as bad learning rate or loss scaling?
- Are gradient norms logged enough to know whether clipping is active?
- Is the threshold defined on raw, accumulated, synchronized, or unscaled gradients?
- Is the clipping ratio logged, not only the threshold?

## Related

- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/observability|Observability]]
