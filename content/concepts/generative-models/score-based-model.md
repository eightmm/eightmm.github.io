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

For a small step in $x$, the score gives the local direction of log-density increase:

$$
\log p_t(x+\Delta x)
\approx
\log p_t(x)
+
s_t(x)^\top \Delta x
$$

This is why a score field can guide samples from low-density noisy states toward likely data states.

For an energy-based density:

$$
p_\theta(x)
\propto
\exp(-E_\theta(x))
$$

the score is:

$$
\nabla_x \log p_\theta(x)
=
-\nabla_x E_\theta(x)
$$

This is why score-based generation, Langevin dynamics, and energy minimization share similar gradient geometry.

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

## Score, Noise, and Clean-Data Targets

For a Gaussian noising process:

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

the conditional score is:

$$
\nabla_{x_t}\log p(x_t\mid x_0)
=
-\frac{\epsilon}{\sigma_t}
$$

So predicting noise, score, or clean data can be converted if the noise convention is known:

$$
s_\theta(x_t,t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sigma_t}
$$

and:

$$
\hat{x}_{0,\theta}
=
\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}
$$

Papers can look different while learning equivalent fields under a shared noising convention. Always identify the parameterization before comparing objectives.

## Langevin View

A score can be used in Langevin dynamics:

$$
x_{k+1}
=
x_k
+
\eta s_\theta(x_k,t)
+
\sqrt{2\eta}\,z_k,
\qquad
z_k\sim\mathcal{N}(0,I)
$$

The gradient term moves toward high-density regions, while the noise term maintains stochastic exploration. Annealed Langevin methods vary $t$ or noise scale from high noise to low noise.

## Sampling

Score-based models can sample with a stochastic reverse-time SDE or a deterministic probability-flow ODE. The choice changes speed, diversity, and likelihood estimation.

| Sampler | Uses | Tradeoff |
| --- | --- | --- |
| reverse SDE | score plus injected noise | stochastic diversity, more sampling choices |
| probability-flow ODE | deterministic score-derived velocity | deterministic samples, likelihood route |
| predictor-corrector | ODE/SDE step plus score correction | quality can improve at higher cost |
| Langevin dynamics | local score ascent plus noise | flexible but step-size sensitive |

## Why It Matters

- Provides a continuous, unifying view of diffusion generation.
- Connects denoising, Langevin dynamics, and SDE/ODE samplers.
- Connects learned density gradients to [[concepts/generative-models/energy-based-model|Energy-based model]] views.
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
- Which target is trained: score, noise, clean data, or velocity?
- Are score scale and noise convention stated?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/energy-based-model|Energy-based model]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
