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

Noise-conditional score matching trains the model across noise levels:

$$
\mathcal{L}
=
\mathbb{E}_{t,x_0,x_t}
\left[
\lambda(t)
\left\|
s_\theta(x_t,t)
- \nabla_{x_t}\log p(x_t\mid x_0)
\right\|_2^2
\right]
$$

For Gaussian perturbations, the target score has a closed form:

$$
\nabla_{x_t}\log p(x_t\mid x_0)
=
-\frac{x_t-\alpha(t)x_0}{\sigma(t)^2}
$$

where $\alpha(t)$ and $\sigma(t)$ define the noising process.

## Sampling

Score-based models can sample with a stochastic reverse-time SDE or a deterministic probability-flow ODE. The choice changes speed, diversity, and likelihood estimation.

## Why It Matters

- Provides a continuous, unifying view of diffusion generation.
- Connects denoising, Langevin dynamics, and SDE/ODE samplers.
- Score estimation quality at all noise levels drives sample quality.

## Failure Modes

- Low-noise scores control detail but are hard to estimate near the data manifold.
- High-noise scores control global structure but can be underweighted.
- Solver choices can dominate quality and cost comparisons.
- The score field may violate geometry or symmetry constraints if the architecture does not encode them.

## Checks

- Is the score well-estimated at both low and high noise?
- SDE vs. ODE sampling for the speed/quality target?
- Does the noise-level weighting match the data?
- Is the sampler, step count, and noise schedule fixed for comparisons?
- Does the score transform correctly under required symmetries?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
