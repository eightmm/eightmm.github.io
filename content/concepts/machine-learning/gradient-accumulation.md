---
title: Gradient Accumulation
tags:
  - machine-learning
  - optimization
  - training
---

# Gradient Accumulation

Gradient accumulation simulates a larger batch by summing gradients over several micro-batches before an optimizer step. It is used when memory cannot fit the desired effective batch size.

For micro-batches $B_1,\ldots,B_A$, the accumulated gradient is:

$$
g_t
=
\frac{1}{A}
\sum_{a=1}^{A}
\nabla_\theta
\mathcal{L}_{B_a}(\theta_t)
$$

Then the optimizer updates once:

$$
\theta_{t+1}
=
\operatorname{Optimizer}(\theta_t, g_t)
$$

where $A$ is the number of accumulation steps.

## Effective Batch Size

With distributed training:

$$
B_{\mathrm{eff}}
=
B_{\mathrm{micro}}
\times
A
\times
N_{\mathrm{devices}}
$$

where $B_{\mathrm{micro}}$ is the per-device micro-batch size and $N_{\mathrm{devices}}$ is the number of devices.

## Why It Matters

- The learning rate schedule usually counts optimizer steps, not micro-batches.
- Gradient clipping may happen per micro-batch or after accumulation; these are different.
- Dropout, batch normalization, padding, and variable-length examples can make accumulated training differ from a true large batch.
- Checkpoint resume should preserve the accumulation position when exact continuation matters.
- Logging should distinguish samples seen, micro-steps, and optimizer steps.

## Checks

- Is the reported batch size micro, per-device, global, or effective?
- Is loss divided by accumulation steps before backpropagation?
- Are gradients zeroed only after the optimizer step?
- Is learning rate warmup based on optimizer steps or consumed samples?
- Is gradient clipping applied at the intended point?
- Does checkpoint state include global step, optimizer step, and accumulation position when needed?
- Are validation and logging intervals counted in examples, micro-steps, or optimizer steps?

## Related

- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
