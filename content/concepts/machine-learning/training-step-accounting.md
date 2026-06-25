---
title: Training Step Accounting
tags:
  - machine-learning
  - optimization
  - training
---

# Training Step Accounting

Training step accounting defines what a "step" means in a run. This matters because [[concepts/machine-learning/training-loop|Training loop]], [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]], [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]], logging, checkpointing, and evaluation intervals often use different counters.

The most useful counters are:

$$
t_{\mathrm{micro}}
=
\text{number of forward/backward micro-batches}
$$

$$
t_{\mathrm{opt}}
=
\text{number of optimizer updates}
$$

$$
n_{\mathrm{seen}}
=
\text{number of training examples or tokens consumed}
$$

With accumulation factor $A$:

$$
t_{\mathrm{opt}}
=
\left\lfloor
\frac{t_{\mathrm{micro}}}{A}
\right\rfloor
$$

and with per-device micro-batch size $B_{\mathrm{micro}}$ and $N$ devices:

$$
n_{\mathrm{seen}}
\approx
t_{\mathrm{micro}}
\times
B_{\mathrm{micro}}
\times
N
$$

For sequence models, token count is often more meaningful than example count:

$$
n_{\mathrm{tokens}}
=
\sum_{b=1}^{t_{\mathrm{micro}}}
\sum_{i\in B_b}
\operatorname{valid\_tokens}(x_i)
$$

## Counters

- Micro-step: one forward/backward pass on one micro-batch.
- Optimizer step: one parameter update after optional accumulation, clipping, scaler update, and optimizer update.
- Scheduler step: usually tied to optimizer steps, but some code ties it to epochs or consumed samples.
- Logging step: the x-axis used in dashboards; it should say whether it is micro-step, optimizer step, sample count, or token count.
- Evaluation step: the interval at which validation is run.
- Checkpoint step: the counter encoded in checkpoint metadata and resume state.

## Canonical Update Boundary

A common deep learning update boundary is:

$$
\mathcal{L}_{b}
=
\frac{1}{A}
\frac{1}{|B_b|}
\sum_{i\in B_b}
\ell(f_\theta(x_i), y_i)
$$

Then each micro-step accumulates:

$$
g
\leftarrow
g + \nabla_\theta \mathcal{L}_{b}
$$

After $A$ micro-steps:

$$
g
\leftarrow
\operatorname{clip}(g)
$$

$$
\theta_{t+1}
=
\operatorname{OptimizerStep}(\theta_t, g, o_t, \eta_t)
$$

$$
o_{t+1}
=
\operatorname{UpdateOptimizerState}(o_t, g)
$$

where $o_t$ is optimizer state and $\eta_t$ is the learning rate at optimizer step $t$.

## Why It Matters

- Warmup length should usually be specified in optimizer steps or consumed tokens, not ambiguous loop iterations.
- Validation every "1000 steps" means different things if accumulation changes.
- Checkpoint resume can drift if the saved state omits accumulation position.
- Throughput claims should state examples/sec or tokens/sec separately from optimizer steps/sec.
- Learning curves become misleading if different runs use different x-axes.

## Checks

- Does the run config define micro-batch size, global batch size, accumulation factor, and effective batch size?
- Is the learning-rate schedule keyed to optimizer steps, epochs, samples, or tokens?
- Are logs labeled with the same counter used by the schedule?
- Does checkpoint metadata include optimizer step, micro-step or consumed samples, epoch, sampler state, and accumulation position?
- Are validation and checkpoint intervals stable when batch size or accumulation changes?
- For variable-length data, is token count tracked separately from example count?

## Related

- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/training-run|Training run]]
