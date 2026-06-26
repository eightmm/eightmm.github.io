---
title: Diffusion Model
tags:
  - diffusion-model
  - generative-model
  - machine-learning
---

# Diffusion Model

A diffusion model learns to reverse a gradual noising process: it is trained to denoise corrupted data and generates samples by iteratively denoising from noise.

A common forward noising form is:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

The model is often trained to predict $\epsilon$, $x_0$, or a related velocity target.

A common noise-prediction objective is:

$$
\mathcal{L}_{\epsilon}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
- \epsilon_\theta(x_t,t)
\right\|_2^2
\right]
$$

The reverse process samples from noise by repeatedly applying a learned denoising transition:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
$$

Alternative parameterizations predict $x_0$ or velocity $v$:

$$
v = \alpha_t \epsilon - \sigma_t x_0
$$

The parameterization affects stability, guidance behavior, and solver design.

## Conditioning

Conditional diffusion adds context $c$:

$$
\epsilon_\theta(x_t,t,c)
$$

Guidance changes the effective score or denoising direction, which can improve fidelity but reduce diversity.

## Why It Matters

- Strong sample quality and stable training across images, molecules, and proteins.
- Flexible conditioning enables guided and inverse-problem generation.
- Iterative sampling is the main cost, motivating faster samplers.

## Failure Modes

- Too few sampling steps can produce invalid or low-detail samples.
- Strong guidance can collapse diversity or overfit the condition.
- The noise schedule may be poorly matched to data scale or geometry.
- Evaluation can confuse validity, novelty, diversity, and downstream utility.

## Checks

- Does the noise schedule fit the data scale?
- Is conditioning/guidance trading diversity for fidelity?
- How many sampling steps are needed for acceptable quality?
- Which prediction target is used: $\epsilon$, $x_0$, score, or velocity?
- Are sampling steps and guidance scales fixed across comparisons?

## Related

- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/consistency-model|Consistency model]]
