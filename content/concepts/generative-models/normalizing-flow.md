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

## Why It Matters

- Exact density evaluation and sampling in one model.
- Useful where calibrated likelihoods matter.
- Invertibility constraints limit architecture choices.

## Checks

- Is the Jacobian determinant tractable for the chosen transforms?
- Is model expressiveness limited by the invertibility constraint?
- Does exact likelihood translate into good sample quality?

## Related

- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
