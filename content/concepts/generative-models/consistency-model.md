---
title: Consistency Model
tags:
  - consistency-model
  - diffusion-model
  - generative-model
---

# Consistency Model

A consistency model maps any point on a diffusion/probability-flow trajectory directly to its origin, enabling one-step or few-step generation.

The self-consistency condition is:

$$
f_\theta(x_t, t)
= f_\theta(x_s, s)
$$

for two points $x_t$ and $x_s$ on the same trajectory. The model learns a time-consistent map back to the clean sample.

A common training loss compares two trajectory points with a stop-gradient target:

$$
\mathcal{L}_{\mathrm{CM}}
=
\mathbb{E}
\left[
d\left(
f_\theta(x_t,t),
\operatorname{sg}(f_{\theta^-}(x_s,s))
\right)
\right]
$$

where $d$ is a distance such as squared error, $\operatorname{sg}$ stops gradients, and $\theta^-$ may be an exponential-moving-average teacher.

## Sampling View

The model learns a map from noisy states to a clean endpoint:

$$
\hat{x}_0 = f_\theta(x_t,t)
$$

This enables one-step sampling from high noise, or few-step sampling by jumping across a small schedule of noise levels.

## Why It Matters

- Targets fast sampling without the full iterative diffusion loop.
- Can be trained from scratch or distilled from a diffusion model.
- Few-step speed must be weighed against sample fidelity.

## Failure Modes

- One-step samples can lose fine detail or diversity.
- Distillation can inherit teacher biases and evaluation gaps.
- Consistency over sparse time pairs may not cover the full trajectory.
- Speed comparisons are misleading if sample filtering or guidance differs.

## Checks

- Is the self-consistency constraint well enforced across the trajectory?
- Distillation from a teacher vs. standalone training?
- How does one-step quality compare to multi-step sampling?
- Is the sampling step count fixed in evaluation?
- Are diversity and validity measured after acceleration?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/sampling|Sampling]]
