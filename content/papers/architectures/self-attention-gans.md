---
title: Self-Attention GAN
aliases:
  - papers/sagan
  - papers/self-attention-gan
  - papers/self-attention-generative-adversarial-networks
  - papers/generative-models/sagan
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - attention
---

# Self-Attention GAN

> The paper inserted self-attention into GANs so image generation could model long-range spatial dependencies.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Self-Attention Generative Adversarial Networks |
| Authors | Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena |
| Year | 2019 |
| Venue | ICML 2019 |
| PMLR | [volume 97 paper](https://proceedings.mlr.press/v97/zhang19d.html) |
| arXiv | [1805.08318](https://arxiv.org/abs/1805.08318) |
| Status | verified |

## Question

Convolutional GANs such as [[papers/architectures/dcgan|DCGAN]] build images through local receptive fields. This is efficient, but a far-away region may only influence another region after many layers.

SAGAN asks:

$$
\text{Can a GAN generator use global spatial context directly?}
$$

The answer is to add self-attention to the generator and discriminator.

## Main Claim

Self-Attention GAN adds non-local attention blocks to image GANs:

$$
h
\rightarrow
h
+
\gamma\operatorname{Attention}(h).
$$

The durable architecture claim is:

$$
\text{convolutional GAN}
+
\text{self-attention over spatial positions}
+
\text{spectral normalization}
\Rightarrow
\text{better long-range image generation}.
$$

This is an architecture paper because it changes how the generator and discriminator exchange information across image locations.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image feature map $x\in\mathbb{R}^{H\times W\times C}$ |
| Attention axis | spatial positions $N=H W$ |
| Query/key/value | learned $1\times1$ projections of feature maps |
| Output | attention-weighted feature map added residually |
| Placement | generator and discriminator feature hierarchy |
| Stabilization | spectral normalization |
| Main bias | long-range dependency modeling for image synthesis |

## Self-Attention Block

Flatten the spatial grid:

$$
x\in\mathbb{R}^{H\times W\times C}
\rightarrow
X\in\mathbb{R}^{N\times C},
\qquad
N=H W.
$$

Compute projections:

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
$$

The attention matrix is:

$$
A
=
\operatorname{softmax}
\left(
QK^\top
\right),
\qquad
A\in\mathbb{R}^{N\times N}.
$$

The attended feature is:

$$
O=AV.
$$

The block returns a residual mixture:

$$
Y
=
X
+
\gamma O,
$$

where $\gamma$ is a learned scalar initialized so the network can start close to the convolutional baseline.

## Why Attention Helps Images

Convolutions are local:

$$
y_{u,v}
=
f(x_{\mathcal{N}(u,v)}).
$$

Self-attention lets a position read from all positions:

$$
y_i
=
\sum_{j=1}^{N}
A_{ij}V_j.
$$

This matters when global consistency is important:

| Visual Need | Why Local Convolution Can Struggle |
| --- | --- |
| object shape consistency | distant parts must agree |
| repeated structures | separated regions share pattern |
| scene layout | global relations affect local detail |
| fine detail conditioned on far context | local receptive field may be insufficient |

## Spectral Normalization

SAGAN also uses spectral normalization to stabilize GAN training. For a weight matrix $W$:

$$
\bar{W}
=
\frac{W}{\sigma(W)},
$$

where $\sigma(W)$ is the largest singular value.

This controls the Lipschitz behavior of layers:

$$
\| \bar{W}x-\bar{W}y \|
\le
\|x-y\|.
$$

In SAGAN, spectral normalization is part of the practical architecture recipe, especially for the discriminator.

## Relation to Transformer Attention

SAGAN attention is not a full Transformer block. It is a non-local spatial block inserted inside a convolutional GAN.

| Model | Attention Role |
| --- | --- |
| [Transformer](/papers/architectures/attention-is-all-you-need) | primary sequence modeling operator |
| SAGAN | global spatial feature mixing inside GAN |
| [DiT](/papers/architectures/scalable-diffusion-models-with-transformers) | primary denoising backbone over latent patches |

The shared mechanism is query-key-value attention; the surrounding architecture is different.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet image generation | attention improves image quality in class-conditional GANs |
| attention maps | generated details can depend on distant regions |
| ablation with attention | gains are tied to adding non-local blocks |
| spectral normalization use | stabilizes adversarial training in the reported setup |

## Limits

- Full spatial attention costs $O(N^2)$ over feature-map positions.
- The result depends on GAN training details, normalization, and conditioning.
- Attention does not remove adversarial training instability.
- Later diffusion and Transformer generators changed the dominant high-resolution generation landscape.

## Concepts

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/generative-models/sampling|Sampling]]

## Related

- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
- [[papers/architectures/dcgan|DCGAN]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/stylegan|StyleGAN]]
