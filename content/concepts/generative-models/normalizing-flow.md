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

## Likelihood Contract

Training usually maximizes exact log likelihood:

$$
\theta^\star
=
\operatorname*{arg\,max}_\theta
\sum_{i=1}^{n}
\log p_\theta(x_i)
$$

This is a density claim. It does not automatically prove sample usefulness, representation quality, or downstream decision quality.

| Claim | Evidence Needed |
| --- | --- |
| exact likelihood | change-of-variables formula and tractable determinant |
| good samples | sampling metrics, validity, diversity, qualitative failures |
| calibrated density | held-out likelihood and OOD likelihood diagnostics |
| useful latent space | interpolation, downstream probe, or controllability evidence |
| fast inference | timing for density evaluation and sampling separately |

## Discrete and Structured Data

Flows are naturally continuous. For discrete objects, the paper should state the relaxation:

| Data Type | Common Handling | Risk |
| --- | --- | --- |
| image pixels | dequantization | likelihood depends on noise model |
| text or SMILES | continuous latent or token relaxation | decoded validity is separate from density |
| molecular graph | graph generation plus continuous relaxation | invertibility and graph validity are hard |
| coordinates | continuous flow in $\mathbb{R}^{N\times 3}$ | equivariance and permutation handling matter |
| protein structure | coordinate or distance flow | residue mapping and constraints matter |

For coordinate data, a plain flow over flattened coordinates can learn frame artifacts unless the symmetry contract is explicit.

## Flow vs ODE Models

Discrete normalizing flows use a finite composition of invertible maps:

$$
x = f_K\circ\cdots\circ f_1(z)
$$

Continuous normalizing flows define:

$$
\frac{dx_t}{dt}=v_\theta(x_t,t)
$$

and adjust density with:

$$
\frac{d\log p_t(x_t)}{dt}
=
-\nabla\cdot v_\theta(x_t,t)
$$

This connects normalizing flows to probability-flow ODEs and flow matching, but the training objective and computational cost can differ.

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
- For coordinates, does the flow respect translation, rotation, and permutation symmetries?
- For molecules or proteins, are invalid decoded samples included in evaluation?

## Related

- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
