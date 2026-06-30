---
title: Glow
aliases:
  - papers/glow
  - papers/generative-flow-with-invertible-1x1-convolutions
  - papers/generative-models/glow
tags:
  - papers
  - architectures
  - generative-models
  - normalizing-flow
  - density-estimation
---

# Glow

> The paper made invertible $1\times1$ convolution a reusable mixing layer for image normalizing flows.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Glow: Generative Flow with Invertible 1x1 Convolutions |
| Authors | Diederik P. Kingma, Prafulla Dhariwal |
| Year | 2018 |
| Venue | NeurIPS 2018 |
| arXiv | [1807.03039](https://arxiv.org/abs/1807.03039) |
| NeurIPS | [paper page](https://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions) |
| Code | [openai/glow](https://github.com/openai/glow) |
| Status | verified |

## Question

[[papers/architectures/real-nvp|Real NVP]] showed that normalizing flows can give exact likelihood, exact sampling, and exact latent inference with affine coupling layers. But coupling layers need a way to mix dimensions between splits.

Glow asks:

$$
\text{Can a flow use a learned invertible channel mixing operation instead of fixed permutations?}
$$

The answer is an invertible $1\times1$ convolution inserted between coupling transformations.

## Main Claim

Glow proposes a flow step:

$$
\text{actnorm}
\rightarrow
\text{invertible }1\times1\text{ convolution}
\rightarrow
\text{affine coupling}.
$$

The durable architecture claim is:

$$
\text{learned invertible channel mixing}
+
\text{coupling layers}
+
\text{multi-scale flow}
\Rightarrow
\text{exact-likelihood image generation with useful latent manipulation}.
$$

This is an architecture paper because the main reusable contribution is a flow block.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor $x\in\mathbb{R}^{H\times W\times C}$ |
| Output | latent tensor $z$ and exact log likelihood $\log p_\theta(x)$ |
| Sampling | draw $z\sim p(z)$, apply inverse flow $x=f_\theta^{-1}(z)$ |
| Core block | actnorm, invertible $1\times1$ convolution, affine coupling |
| Main guarantee | exact inverse and tractable log-determinant |
| Main bias | image density modeling through invertible transformations |

## Change of Variables

Glow keeps the normalizing-flow contract:

$$
z = f_\theta(x),
\qquad
x = f_\theta^{-1}(z).
$$

For an invertible map:

$$
\log p_X(x)
=
\log p_Z(f_\theta(x))
+
\log
\left|
\det
\frac{\partial f_\theta(x)}{\partial x}
\right|.
$$

For a stack of flow steps:

$$
\log p_X(x)
=
\log p_Z(z_K)
+
\sum_{k=1}^{K}
\log
\left|
\det
\frac{\partial z_k}{\partial z_{k-1}}
\right|.
$$

The architecture must therefore make both inverse computation and log-determinant computation tractable.

## Flow Step

Glow repeats three transformations.

| Component | Transformation | Why It Exists |
| --- | --- | --- |
| actnorm | per-channel affine normalization | data-dependent initialization without batch statistics |
| invertible $1\times1$ conv | learned channel mixing | replaces fixed channel permutation |
| affine coupling | expressive invertible transform | nonlinear density modeling with easy inverse |

The step can be written as:

$$
h_1 = \operatorname{ActNorm}(h_0),
$$

$$
h_2 = \operatorname{Conv}_{1\times1}^{\mathrm{inv}}(h_1),
$$

$$
h_3 = \operatorname{Coupling}(h_2).
$$

Each component has a computable inverse.

## Invertible 1x1 Convolution

At each spatial location, Glow applies the same learned invertible matrix:

$$
y_{u,v} = W x_{u,v},
\qquad
W\in\mathbb{R}^{C\times C}.
$$

Here $x_{u,v}$ is the channel vector at pixel location $(u,v)$.

The inverse is:

$$
x_{u,v} = W^{-1} y_{u,v}.
$$

The log-determinant contribution is:

$$
\log
\left|
\det
\frac{\partial y}{\partial x}
\right|
=
H\cdot W_{\text{spatial}}
\log |\det W|.
$$

Using $W_{\text{spatial}}$ for image width avoids confusing it with the convolution matrix $W$.

In practice, the matrix can be parameterized with LU decomposition to make determinant and inverse computation cheaper.

## Why It Improves Real NVP

Real NVP typically relies on fixed permutations or masks so that different dimensions eventually influence each other. Glow makes this mixing learned:

| Axis | Real NVP | Glow |
| --- | --- | --- |
| dimension mixing | fixed permutations/masks | learned invertible $1\times1$ convolution |
| normalization | batchnorm-like choices | actnorm |
| coupling | affine coupling | affine coupling |
| likelihood | exact | exact |
| inverse | exact | exact |

The learned channel mixing is the main architectural addition.

## Multi-Scale Flow

Glow follows the multi-scale image-flow pattern:

$$
x
\rightarrow
\text{squeeze}
\rightarrow
\text{flow steps}
\rightarrow
\text{split}
\rightarrow
\cdots
\rightarrow
z.
$$

The squeeze operation trades spatial resolution for channels:

$$
H\times W\times C
\rightarrow
\frac{H}{2}\times\frac{W}{2}\times 4C.
$$

This lets channel mixing operate on local spatial neighborhoods after reshaping.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| likelihood benchmarks | invertible $1\times1$ convolution improves flow density modeling |
| large-image samples | likelihood-trained flows can generate visually plausible high-resolution samples |
| latent interpolation and manipulation | exact latent inference gives a controllable representation route |

The evidence should be read as a flow-architecture claim, not as a claim that likelihood alone always optimizes perceptual quality.

## Limits

- Invertibility constrains architecture design.
- Exact likelihood can prefer density behavior that does not perfectly match human perceptual quality.
- Sampling is parallel through the inverse, but memory and determinant costs still matter.
- Image flows have become less dominant than diffusion models for high-fidelity generation, but Glow remains central for understanding invertible architectures.

## Concepts

- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/latent-variable-model|Latent-variable model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/computational-complexity|Computational complexity]]

## Related

- [[papers/architectures/real-nvp|Real NVP]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/ddpm|Denoising Diffusion Probabilistic Models]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
