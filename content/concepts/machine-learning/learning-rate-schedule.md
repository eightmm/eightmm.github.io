---
title: Learning Rate Schedule
tags:
  - machine-learning
  - optimization
---

# Learning Rate Schedule

A learning rate schedule changes the update size during training. It can determine whether training is stable, fast, or brittle.

The update rule usually uses the learning rate at optimizer step $t$:

$$
\theta_{t+1}
=
\theta_t
-
\eta_t
u_t
$$

where $u_t$ is the optimizer-specific update direction. The schedule defines $\eta_t$.

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

Piecewise schedules make the step boundary explicit:

$$
\eta_t
=
\begin{cases}
\eta_{\max}\frac{t}{T_{\mathrm{warmup}}}, & t < T_{\mathrm{warmup}} \\
\eta_{\max}, & T_{\mathrm{warmup}} \le t < T_{\mathrm{decay}} \\
\eta_{\min}, & t \ge T_{\mathrm{decay}}
\end{cases}
$$

The variable $t$ should be defined by [[concepts/machine-learning/training-step-accounting|training step accounting]]. In most deep learning code, $t$ is the optimizer step, not the micro-step.

## Checks

- Is warmup needed for large batch, mixed precision, or unstable early training?
- Is total step count defined before choosing the schedule?
- Are steps counted as micro-steps, optimizer steps, consumed samples, or epochs?
- Does changing accumulation keep the intended number of optimizer steps or consumed samples?
- Are scheduler state and current step saved in checkpoints?
- Are comparisons fair when different methods use different schedules?

## Related

- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[infra/reproducibility/run-record|Reproducible run record]]
