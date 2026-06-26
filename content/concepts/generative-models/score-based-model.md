---
title: Score-Based Model
tags:
  - score-based-model
  - diffusion-model
  - generative-model
---

# Score-Based Model

A score-based model learns the gradient of the log data density (the score) across noise levels and samples by following the score, often via a stochastic differential equation.

The score is:

$$
s_\theta(x,t)
\approx \nabla_x \log p_t(x)
$$

It points toward regions of higher probability under the noisy distribution at time $t$.

## Why It Matters

- Provides a continuous, unifying view of diffusion generation.
- Connects denoising, Langevin dynamics, and SDE/ODE samplers.
- Score estimation quality at all noise levels drives sample quality.

## Checks

- Is the score well-estimated at both low and high noise?
- SDE vs. ODE sampling for the speed/quality target?
- Does the noise-level weighting match the data?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
