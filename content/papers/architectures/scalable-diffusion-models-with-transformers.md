---
title: Scalable Diffusion Models with Transformers
aliases:
  - papers/dit
  - papers/diffusion-transformer
  - papers/scalable-diffusion-models-with-transformers
  - papers/generative-models/dit
tags:
  - papers
  - architectures
  - generative-models
  - diffusion
  - transformer
---

# Scalable Diffusion Models with Transformers

> The paper replaces the usual convolutional U-Net denoiser in latent diffusion with a Transformer over latent image patches.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Scalable Diffusion Models with Transformers |
| Authors | William Peebles, Saining Xie |
| Year | 2022 preprint; 2023 conference |
| Venue | ICCV 2023 |
| arXiv | [2212.09748](https://arxiv.org/abs/2212.09748) |
| Project | [DiT project page](https://www.wpeebles.com/DiT.html) |
| Code | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) |
| Status | verified |

## Question

[[papers/architectures/ddpm|DDPM]] and [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]] often use convolutional U-Net denoisers. DiT asks:

$$
\text{Can a Transformer be the main denoising backbone for image diffusion?}
$$

The paper's answer:

$$
\text{latent image}
\rightarrow
\text{patch tokens}
\rightarrow
\text{Transformer denoiser}
\rightarrow
\text{noise or velocity prediction}.
$$

## Main Claim

Diffusion Transformers replace the U-Net backbone with a Transformer operating on latent patches and show predictable scaling with model compute.

The durable architecture claim is:

$$
\text{latent diffusion}
+
\text{ViT-style patch tokens}
+
\text{conditional Transformer blocks}
\Rightarrow
\text{scalable diffusion backbone}.
$$

This is an architecture paper because it changes the denoising network family, not the basic diffusion objective.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | noisy latent image $z_t\in\mathbb{R}^{h\times w\times c}$ |
| Tokenization | split latent into patches |
| Backbone | Transformer blocks |
| Conditioning | timestep and class conditioning |
| Output | predicted noise, velocity, or denoising target over latent patches |
| Decoder | latent decoder maps denoised latent to image |
| Main comparison | Transformer denoiser vs U-Net denoiser |
| Scaling axis | depth, width, patch size, token count, Gflops |

## Latent Patch Tokenization

DiT operates in latent space. Given:

$$
z_t\in\mathbb{R}^{h\times w\times c},
$$

split it into patches of size $p\times p$:

$$
N
=
\frac{h w}{p^2}.
$$

Each patch is projected into a token:

$$
x_i = W_{\text{patch}}\operatorname{vec}(z_{t,i}) + b.
$$

The Transformer receives:

$$
X = [x_1,\dots,x_N] + P,
$$

where $P$ is positional information.

Smaller patch size gives more tokens:

$$
p\downarrow
\Rightarrow
N\uparrow
\Rightarrow
\text{more compute and potentially better quality}.
$$

## Transformer Denoiser

A DiT block is a Transformer block applied to latent patch tokens:

$$
H_{\ell+1}
=
\operatorname{Block}_{\ell}(H_\ell, t, y).
$$

The self-attention part follows the usual form:

$$
Q = H W_Q,
\qquad
K = H W_K,
\qquad
V = H W_V,
$$

$$
\operatorname{Attention}(H)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

The key change from U-Net diffusion is the spatial mixing operator:

| Backbone | Spatial Mixing |
| --- | --- |
| U-Net denoiser | convolution, down/up sampling, skip paths |
| DiT denoiser | global self-attention over latent patch tokens |

## Conditioning

Diffusion denoisers need timestep conditioning. Class-conditional image generation also needs label conditioning.

DiT studies conditioning mechanisms, with adaptive layer normalization as a central route. Abstractly:

$$
c = g(t,y),
$$

where $t$ is timestep and $y$ is optional class label.

The conditioned normalization can be written:

$$
\operatorname{adaLN}(h,c)
=
\gamma(c)\odot
\frac{h-\mu(h)}{\sigma(h)}
+
\beta(c).
$$

This lets the conditioning signal modulate the Transformer block without changing the token sequence length.

## Output Head

After Transformer blocks, tokens are mapped back to latent patches:

$$
\hat{\epsilon}_{i}
=
W_{\text{out}} h_i + b_{\text{out}}.
$$

The patches are unpatchified:

$$
\{\hat{\epsilon}_i\}_{i=1}^{N}
\rightarrow
\hat{\epsilon}_\theta(z_t,t,y)
\in
\mathbb{R}^{h\times w\times c}.
$$

The diffusion loss can use the standard noise-prediction form:

$$
\mathcal{L}
=
\mathbb{E}_{z_0,\epsilon,t,y}
\left[
\left\|
\epsilon
-
\epsilon_\theta(z_t,t,y)
\right\|_2^2
\right].
$$

## Scaling View

DiT evaluates scaling through forward-pass compute:

$$
\text{quality}
\approx
f(\text{Gflops}).
$$

Compute can increase by:

| Axis | Effect |
| --- | --- |
| depth | more Transformer blocks |
| width | larger hidden dimension |
| patch size $p$ | smaller $p$ gives more tokens |
| token count $N$ | attention and MLP cost increase |

The important empirical claim is that increasing DiT compute through these axes consistently improves FID in the tested ImageNet setting.

## Why It Matters

DiT is important because it moves the architecture conversation from:

$$
\text{diffusion model}
=
\text{U-Net denoiser}
$$

to:

$$
\text{diffusion model}
=
\text{objective}
+
\text{replaceable denoising backbone}.
$$

This opened the path for Transformer-heavy image/video diffusion backbones where scale, tokenization, and conditioning are first-class design variables.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet class-conditional generation | Transformer denoisers can outperform prior diffusion models under large scale |
| scaling curves | higher Gflops correlate with lower FID in the tested setup |
| patch-size comparisons | token count is a meaningful scaling route |
| conditioning ablations | block conditioning design matters for diffusion Transformers |

## Limits

- DiT is evaluated mainly in class-conditional ImageNet latent diffusion.
- Transformer attention cost grows with token count.
- The scaling claim is tied to compute, data, training recipe, and benchmark setting.
- U-Net backbones still remain strong in many settings, especially when locality and multi-scale skip structure are valuable.

## Concepts

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]

## Related

- [[papers/architectures/ddpm|Denoising Diffusion Probabilistic Models]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
