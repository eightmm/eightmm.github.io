---
title: Normalizing Flow
tags:
  - normalizing-flow
  - generative-model
  - machine-learning
---

# Normalizing Flow

A normalizing flow transforms a simple base distribution into a complex one through a sequence of invertible, differentiable maps, giving exact likelihoods via the change-of-variables formula.

The change-of-variables formula is:

$$
\log p_X(x)
= \log p_Z(f^{-1}(x))
+ \log \left|
\det \frac{\partial f^{-1}}{\partial x}
\right|
$$

The transform must be invertible and have a tractable Jacobian determinant.

For a sequence of invertible maps $f = f_K \circ \cdots \circ f_1$:

$$
z_0 \sim p_Z(z), \qquad
z_k = f_k(z_{k-1}), \qquad
x = z_K
$$

and the log density is:

$$
\log p_X(x)
=
\log p_Z(z_0)
-
\sum_{k=1}^{K}
\log \left|
\det \frac{\partial f_k}{\partial z_{k-1}}
\right|
$$

where $z_0 = f^{-1}(x)$.

## Design Constraint

The architecture must balance expressiveness and tractable determinants. Coupling layers, autoregressive flows, invertible $1\times1$ convolutions, and continuous-time flows make different tradeoffs between parallel sampling, parallel density evaluation, and transform flexibility.

## Why It Matters

- Exact density evaluation and sampling in one model.
- Useful where calibrated likelihoods matter.
- Invertibility constraints limit architecture choices.

## Failure Modes

- High likelihood can still produce poor perceptual or task-level samples.
- Invertibility can force the latent and data dimensions to match or require dequantization.
- Tractable Jacobians can restrict interaction between dimensions.
- Out-of-distribution data can receive misleading likelihoods.

## Checks

- Is the Jacobian determinant tractable for the chosen transforms?
- Is model expressiveness limited by the invertibility constraint?
- Does exact likelihood translate into good sample quality?
- Are likelihood and downstream utility evaluated separately?
- Does the model need discrete dequantization or a continuous relaxation?

## Related

- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
