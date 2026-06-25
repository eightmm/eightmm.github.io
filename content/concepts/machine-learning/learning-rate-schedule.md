---
title: Learning Rate Schedule
tags:
  - machine-learning
  - optimization
---

# Learning Rate Schedule

A learning rate schedule changes the update size during training. It can determine whether training is stable, fast, or brittle.

A simple warmup schedule is:

$$
\eta_t
=
\eta_{\max}
\min\left(1,\frac{t}{T_{\mathrm{warmup}}}\right)
$$

A cosine decay schedule is:

$$
\eta_t
=
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(
1+\cos\frac{\pi t}{T}
\right)
$$

## Checks

- Is warmup needed for large batch, mixed precision, or unstable early training?
- Is total step count defined before choosing the schedule?
- Are scheduler state and current step saved in checkpoints?
- Are comparisons fair when different methods use different schedules?

## Related

- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[infra/reproducible-run-record|Reproducible run record]]
