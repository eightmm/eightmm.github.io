---
title: Score Matching
tags:
  - generative-models
  - score-based-model
  - diffusion
---

# Score Matching

Score matching trains a model to estimate the score function of a probability distribution:

$$
s_\theta(x) \approx \nabla_x \log p_{\mathrm{data}}(x)
$$

The score points in the direction where the log density increases. In diffusion and score-based models, the score is learned across noise levels:

$$
s_\theta(x_t,t) \approx \nabla_{x_t}\log p_t(x_t)
$$

$p_t$ is the noisy data distribution at time or noise level $t$.

## Denoising Score Matching

When clean data $x_0$ is corrupted by Gaussian noise:

$$
x_t = x_0 + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

the denoising target is proportional to the noise:

$$
\nabla_{x_t}\log p(x_t \mid x_0)
=
-\frac{x_t-x_0}{\sigma_t^2}
=
-\frac{\epsilon}{\sigma_t}
$$

This connects score prediction, denoising, and noise prediction.

## Why It Matters

- Gives diffusion models a density-gradient interpretation.
- Makes sampling possible with Langevin dynamics, reverse SDEs, or probability-flow ODEs.
- Separates the learned object from a particular sampler.
- Helps explain why noise-level weighting changes sample quality.

## Checks

- Which score is learned: clean data, noisy marginal, conditional score, or classifier guidance score?
- Are low-noise and high-noise regions weighted appropriately?
- Does the target preserve the symmetry of the object, such as permutation or rotation equivariance?
- Is the score evaluated only through samples, or also through likelihood or reconstruction diagnostics?

## Related

- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/math/normal-distribution|Normal distribution]]
