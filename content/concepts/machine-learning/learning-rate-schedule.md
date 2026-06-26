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

## Schedule Contract

| Field | Question |
| --- | --- |
| Step unit | optimizer step, micro-step, sample, token, or epoch? |
| Warmup | number of steps and maximum LR |
| Decay | cosine, linear, exponential, step, constant, WSD, or custom |
| Minimum LR | final learning rate or zero? |
| Parameter groups | same schedule for all parameters or separate groups? |
| Resume state | scheduler step saved in checkpoint? |

Record the total training budget with the schedule:

$$
N_{\mathrm{tokens}}
=
N_{\mathrm{steps}}
\times
B_{\mathrm{global}}
\times
L_{\mathrm{seq}}
$$

for token models, or the matching sample/object count for non-text data.

## Common Patterns

| Pattern | Formula Sketch | Use |
| --- | --- | --- |
| constant | $\eta_t=\eta$ | small or stable runs |
| linear warmup | $\eta_t=\eta_{\max}t/T_w$ | avoid unstable early updates |
| cosine decay | smooth decay to $\eta_{\min}$ | common deep learning default |
| linear decay | $\eta_t=\eta_{\max}(1-t/T)$ | simple finite-budget training |
| step decay | multiply by factor at milestones | classical training recipes |
| WSD | warmup, stable plateau, decay | long pretraining-style runs |

Different schedules can change the effective training recipe even when architecture and dataset are unchanged.

## Batch and Accumulation Coupling

If gradient accumulation changes, the number of optimizer steps can change unless explicitly controlled:

$$
B_{\mathrm{global}}
=
B_{\mathrm{micro}}
\times
N_{\mathrm{devices}}
\times
N_{\mathrm{accum}}
$$

The scheduler usually steps once per optimizer update:

$$
t_{\mathrm{sched}} = t_{\mathrm{opt}}
$$

not once per micro-batch. This distinction matters for reproducibility and paper comparisons.

## Paper Reading Risks

| Risk | Why It Matters |
| --- | --- |
| unspecified warmup | early instability or unfair comparison |
| different total steps | more compute hidden as better method |
| schedule tuned per baseline | baseline may be under-optimized |
| missing resume state | resumed runs follow a different LR curve |
| separate LR groups | backbone, head, embeddings, or adapters learn at different speeds |

## Checks

- Is warmup needed for large batch, mixed precision, or unstable early training?
- Is total step count defined before choosing the schedule?
- Are steps counted as micro-steps, optimizer steps, consumed samples, or epochs?
- Does changing accumulation keep the intended number of optimizer steps or consumed samples?
- Are scheduler state and current step saved in checkpoints?
- Are comparisons fair when different methods use different schedules?
- Are reported results tied to compute budget, consumed samples/tokens, and schedule?
- Are baselines using comparable tuning effort and schedule families?

## Related

- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[infra/reproducibility/run-record|Reproducible run record]]
