---
title: StyleGAN2
aliases:
  - papers/stylegan2
  - papers/analyzing-and-improving-stylegan
  - papers/generative-models/stylegan2
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - image-generation
---

# StyleGAN2

> The paper redesigned StyleGAN's normalization and regularization to reduce artifacts and improve image quality.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Analyzing and Improving the Image Quality of StyleGAN |
| Authors | Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila |
| Year | 2019 preprint; 2020 conference |
| Venue | CVPR 2020 |
| arXiv | [1912.04958](https://arxiv.org/abs/1912.04958) |
| CVF | [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html) |
| Code | [NVlabs/stylegan2](https://github.com/NVlabs/stylegan2) |
| Status | verified |

## Question

[[papers/architectures/stylegan|StyleGAN]] made style-based synthesis a canonical high-resolution GAN architecture, but it also introduced characteristic artifacts. StyleGAN2 asks:

$$
\text{Which parts of StyleGAN's generator architecture cause artifacts, and how should they be redesigned?}
$$

The answer is a revised generator with weight modulation/demodulation, no progressive growing, and path length regularization.

## Main Claim

StyleGAN2 keeps the style-based synthesis idea but changes how styles affect convolutional weights and how the generator is regularized.

The durable architecture claim is:

$$
\text{style modulation}
+
\text{weight demodulation}
+
\text{skip/residual generator redesign}
+
\text{path length regularization}
\Rightarrow
\text{cleaner high-resolution GAN synthesis}.
$$

This is not just a training tweak. It is a generator-architecture correction paper.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | latent code $z$ mapped to intermediate style $w$ |
| Generator | style-based convolutional synthesis network |
| Main change | replace AdaIN-style feature normalization with weight modulation and demodulation |
| Training topology | avoid progressive growing in favor of a fixed architecture |
| Regularization | path length regularization for smoother latent-to-image mapping |
| Output | high-resolution image |
| Main target | reduce StyleGAN artifacts and improve image quality |

## From AdaIN to Weight Demodulation

StyleGAN injects style through adaptive instance normalization:

$$
\operatorname{AdaIN}(x,y)
=
y_s
\frac{x-\mu(x)}{\sigma(x)}
+
y_b.
$$

StyleGAN2 argues that this feature normalization can contribute to artifacts. The redesigned view modulates convolution weights directly.

Let $w$ be a style vector and let $s_i$ be a per-input-channel style scale. For convolution weight:

$$
W_{ijk},
$$

where $i$ indexes output channel, $j$ input channel, and $k$ spatial kernel position, modulation gives:

$$
W'_{ijk}=s_j W_{ijk}.
$$

Demodulation normalizes each output channel:

$$
W''_{ijk}
=
\frac{W'_{ijk}}
{\sqrt{\sum_{j,k}(W'_{ijk})^2+\epsilon}}.
$$

This preserves style control while reducing signal magnitude artifacts.

## Why Remove Progressive Growing?

[[papers/architectures/progressive-growing-of-gans|Progressive GAN]] grows the network resolution during training:

$$
4^2\rightarrow 8^2\rightarrow\cdots\rightarrow1024^2.
$$

StyleGAN inherited this lineage. StyleGAN2 revisits it because progressive growing can leave resolution-dependent artifacts and makes the generator/discriminator interface change during training.

The revised approach uses a fixed generator architecture:

$$
\text{all resolutions exist from the start}.
$$

This makes training topology more stable and easier to analyze.

## Path Length Regularization

The mapping from latent $w$ to image should have a well-conditioned Jacobian. StyleGAN2 encourages small and consistent changes in image space for changes in latent space.

Let:

$$
G(w)
\in
\mathbb{R}^{H\times W\times C}.
$$

The local sensitivity is related to:

$$
J_w
=
\frac{\partial G(w)}{\partial w}.
$$

Path length regularization penalizes deviations in the norm of image-space changes induced by latent perturbations:

$$
\mathcal{L}_{\text{pl}}
=
\mathbb{E}_{w,y}
\left(
\left\|
J_w^\top y
\right\|_2
-
a
\right)^2,
$$

where $y$ is a random image-space direction and $a$ is a moving average target length.

The practical effect is smoother latent interpolation and easier generator inversion.

## Generator Redesign

StyleGAN2 compares generator structures and uses skip or residual paths rather than relying on progressive growing.

| Component | StyleGAN | StyleGAN2 |
| --- | --- | --- |
| style injection | AdaIN on activations | weight modulation/demodulation |
| training topology | progressive growing | fixed architecture |
| artifact diagnosis | observed after synthesis | explicitly analyzed and targeted |
| latent smoothness | style mixing and metrics | path length regularization |

## Relation to StyleGAN Lineage

| Paper | Main Contribution |
| --- | --- |
| [Progressive GAN](/papers/architectures/progressive-growing-of-gans) | grow generator/discriminator resolution over training |
| [StyleGAN](/papers/architectures/stylegan) | style-based per-layer synthesis control |
| StyleGAN2 | weight demodulation and fixed generator redesign to reduce artifacts |

The sequence is useful because it shows how high-resolution GANs moved from training curriculum to generator control to artifact-aware architecture design.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| artifact analysis | StyleGAN artifacts are tied to architecture/normalization behavior |
| image quality metrics | revised generator improves distribution quality metrics |
| perceptual comparisons | artifacts are reduced in generated images |
| path length regularization analysis | latent interpolation and inversion behavior improve |

## Limits

- The paper is centered on unconditional image generation.
- It improves the StyleGAN lineage but does not remove adversarial training tradeoffs.
- The architecture is still compute- and data-sensitive.
- Later StyleGAN variants address aliasing, data efficiency, and domain adaptation separately.

## Concepts

- [[concepts/generative-models/gan|GAN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]

## Related

- [[papers/architectures/stylegan|StyleGAN]]
- [[papers/architectures/progressive-growing-of-gans|Progressive Growing of GANs]]
- [[papers/architectures/biggan|BigGAN]]
- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
