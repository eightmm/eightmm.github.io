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

## Training Objective

A common denoising score matching objective is:

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\lambda(t)
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log p(x_t\mid x_0)
\right\|_2^2
\right]
$$

Using the Gaussian corruption above:

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\lambda(t)
\left\|
s_\theta(x_t,t)
+
\frac{\epsilon}{\sigma_t}
\right\|_2^2
\right]
$$

The weighting $\lambda(t)$ controls which noise levels dominate training. This is not a cosmetic detail; it changes sample quality, likelihood behavior, and low-noise reconstruction.

## Score, Noise, and Denoised Prediction

Many diffusion papers parameterize the same target in different ways:

| Parameterization | Model Predicts | Conversion Idea |
| --- | --- | --- |
| score | $\nabla_{x_t}\log p_t(x_t)$ | density-gradient field |
| noise | $\epsilon$ | score is proportional to $-\epsilon/\sigma_t$ |
| clean data | $x_0$ | denoised estimate determines score under Gaussian corruption |
| velocity | transport or interpolation velocity | common in flow/rectified formulations |

When reading a paper, identify the parameterization before comparing objectives.

## Geometry and Symmetry

For coordinate data, the score has the same transformation type as coordinates. If:

$$
X' = XR^\top + \mathbf{1}t^\top
$$

then an equivariant score should satisfy:

$$
s_\theta(X',t)=s_\theta(X,t)R^\top
$$

For molecules, proteins, and complexes, the score target also depends on atom/residue mapping, masking, coordinate frame, and whether noise is added to coordinates, features, or both.

## Sampling Boundary

Learning a score is not the same as specifying the final sampler. Samples may come from Langevin dynamics, reverse SDEs, predictor-corrector methods, or probability-flow ODEs. Report the number of function evaluations, noise schedule, guidance, and filtering when comparing generation quality.

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
- What parameterization is used: score, noise, clean data, or velocity?
- What sampler and step budget turn the learned field into samples?

## Related

- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/math/normal-distribution|Normal distribution]]
