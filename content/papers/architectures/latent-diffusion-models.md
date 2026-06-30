---
title: Latent Diffusion Models
aliases:
  - papers/latent-diffusion-models
  - papers/ldm
  - papers/stable-diffusion
  - papers/high-resolution-image-synthesis-with-latent-diffusion-models
tags:
  - papers
  - architectures
  - generative-models
  - diffusion
---

# Latent Diffusion Models

> The paper moves diffusion modeling from pixel space into the latent space of a pretrained autoencoder and adds cross-attention conditioning for flexible high-resolution generation.

## Metadata

| Field | Value |
| --- | --- |
| Paper | High-Resolution Image Synthesis with Latent Diffusion Models |
| Authors | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer |
| Year | 2021 preprint; 2022 conference |
| Venue | CVPR 2022 |
| arXiv | [2112.10752](https://arxiv.org/abs/2112.10752) |
| CVF | [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
| Code | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) |
| Status | full note started |

## One-Line Takeaway

Latent Diffusion Models split image generation into:

$$
x
\xrightarrow{\mathcal{E}}
z
\xrightarrow{\text{diffusion in latent space}}
\hat{z}
\xrightarrow{\mathcal{D}}
\hat{x},
$$

where $\mathcal{E}$ and $\mathcal{D}$ are a pretrained image autoencoder, and the diffusion model operates on compressed latent $z$ rather than pixels $x$.

## Question

Pixel-space diffusion models are high quality, but expensive. For an image:

$$
x\in\mathbb{R}^{H\times W\times 3},
$$

the denoising U-Net must run repeatedly over a large spatial tensor.

The architecture question is:

$$
\text{Can diffusion keep image quality while denoising a smaller representation?}
$$

LDM answers by choosing a perceptual latent space:

$$
z=\mathcal{E}(x)
\in
\mathbb{R}^{h\times w\times c},
\qquad
h=\frac{H}{f},\quad w=\frac{W}{f}.
$$

The diffusion model works in $z$-space, then a decoder maps back to pixels.

## Main Claim

Diffusion in a learned latent space gives a better compute-quality tradeoff than diffusion directly in pixels, while cross-attention turns the denoiser into a flexible conditional generator.

The compact claim:

$$
\text{perceptual compression}
+
\text{latent diffusion}
+
\text{cross-attention conditioning}
\Rightarrow
\text{efficient high-resolution synthesis}.
$$

This is an architecture claim because it changes where the generative process happens and how conditioning enters the denoising network.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input image | $x\in\mathbb{R}^{H\times W\times 3}$ |
| Encoder | $\mathcal{E}:x\mapsto z$ |
| Latent | $z\in\mathbb{R}^{h\times w\times c}$ |
| Compression factor | $f=H/h=W/w$ |
| Generator | denoising diffusion model in latent space |
| Backbone | U-Net-style denoiser |
| Conditioning | cross-attention over text, layout, masks, boxes, or other context |
| Decoder | $\mathcal{D}:z\mapsto x$ |
| Output | image or image-like prediction |
| Main benefit | lower compute and memory than pixel diffusion |

## Autoencoder Stage

The first stage learns an image autoencoder:

$$
z=\mathcal{E}(x),
\qquad
\hat{x}=\mathcal{D}(z).
$$

The autoencoder should compress spatial resolution while preserving perceptual detail:

$$
\mathcal{D}(\mathcal{E}(x))\approx x.
$$

The key architecture choice is the compression factor $f$:

| Compression | Effect |
| --- | --- |
| small $f$ | high fidelity, more diffusion compute |
| large $f$ | cheaper diffusion, more reconstruction loss |
| moderate $f$ | target tradeoff in LDM |

If compression is too weak, latent diffusion remains expensive. If compression is too aggressive, the decoder cannot reconstruct high-quality images.

## Latent Diffusion Objective

After training the autoencoder, diffusion is applied to the latent representation:

$$
z_0=\mathcal{E}(x).
$$

The forward noising process is:

$$
q(z_t\mid z_0)
=
\mathcal{N}
\left(
z_t;
\sqrt{\bar{\alpha}_t}z_0,
(1-\bar{\alpha}_t)I
\right).
$$

Equivalently:

$$
z_t
=
\sqrt{\bar{\alpha}_t}z_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

The denoiser predicts noise:

$$
\epsilon_\theta(z_t,t,c),
$$

where $c$ is optional conditioning.

The simplified training loss is:

$$
\mathcal{L}_{\text{LDM}}
=
\mathbb{E}_{z_0,\epsilon,t,c}
\left[
\left\|
\epsilon
-
\epsilon_\theta(z_t,t,c)
\right\|_2^2
\right].
$$

This is the [[papers/architectures/ddpm|DDPM]] objective moved from $x$ to $z$.

## Why Latent Space Helps

If the autoencoder downsamples by factor $f$, the spatial area shrinks by:

$$
f^2.
$$

For example, $512\times512$ pixels with $f=8$ becomes:

$$
64\times64
$$

latent spatial resolution.

The denoising U-Net therefore runs on a much smaller tensor:

$$
H W
\rightarrow
\frac{HW}{f^2}.
$$

The exact compute does not shrink only by $f^2$ because channel count, attention blocks, and decoder cost matter. But the dominant spatial denoising workload is greatly reduced.

## Cross-Attention Conditioning

LDM adds conditioning through cross-attention layers inside the denoising U-Net.

Let $h$ be an intermediate latent feature map flattened into spatial tokens, and:

$$
y=\tau_\phi(c)
$$

be conditioning tokens from a condition encoder, such as text embeddings.

Cross-attention computes:

$$
Q = hW_Q,
\qquad
K = yW_K,
\qquad
V = yW_V.
$$

Then:

$$
\operatorname{CrossAttn}(h,y)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

This lets each spatial latent location attend to condition tokens.

The important contract:

$$
\text{condition encoder}
\rightarrow
\text{condition tokens}
\rightarrow
\text{cross-attention in denoiser}.
$$

This is why LDM can support text, class labels, segmentation maps, bounding boxes, masks, and other conditioning forms under a shared architecture pattern.

## Sampling Path

At sampling time:

1. start from latent noise:

$$
z_T\sim\mathcal{N}(0,I);
$$

2. run iterative denoising:

$$
z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_0;
$$

3. decode:

$$
\hat{x}=\mathcal{D}(z_0).
$$

The generated image is only as good as both parts:

$$
\text{diffusion prior quality}
\quad
\text{and}
\quad
\text{decoder reconstruction quality}.
$$

## Relation To DDPM

[[papers/architectures/ddpm|DDPM]] gives the core denoising diffusion framework:

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon.
$$

LDM keeps the diffusion idea but changes the data space:

$$
x_t
\quad
\text{becomes}
\quad
z_t.
$$

| Axis | Pixel Diffusion | Latent Diffusion |
| --- | --- | --- |
| denoising space | pixels | autoencoder latent |
| compute | high at large resolution | lower due to spatial compression |
| output detail | directly modeled in pixels | depends on decoder |
| conditioning | possible | cross-attention emphasized |
| high-resolution synthesis | expensive | more practical |

LDM should not be read as replacing diffusion. It is a better architectural placement for the diffusion process.

## Relation To VAE And Autoencoders

The autoencoder in LDM is not the whole generative model. It provides a latent space where the diffusion prior operates.

Compare:

| Model | Latent Role |
| --- | --- |
| [[papers/architectures/auto-encoding-variational-bayes|VAE]] | probabilistic latent variable model trained end-to-end with ELBO |
| VQGAN-like autoencoder | compress image into perceptual latent or code representation |
| LDM | trains diffusion prior over a pretrained perceptual latent |

The key difference:

$$
\text{LDM generation}
\neq
\text{sample from simple Gaussian latent then decode}.
$$

LDM samples by an iterative denoising process in latent space.

## Relation To Stable Diffusion

Stable Diffusion is a prominent text-to-image system built from the LDM architecture family:

$$
\text{text encoder}
\rightarrow
\text{cross-attention conditioned latent denoiser}
\rightarrow
\text{image decoder}.
$$

The paper note should not be reduced to one product. The reusable architecture idea is:

$$
\text{do expensive generative modeling in a learned perceptual latent}.
$$

Later systems modify text encoders, U-Net sizes, guidance, data curation, conditioning, samplers, and training recipes, but the LDM decomposition remains a durable pattern.

## Why It Belongs In Architecture Papers

LDM is a canonical architecture paper because it changes the location of the generative model:

$$
\mathcal{X}_{\text{pixel}}
\rightarrow
\mathcal{Z}_{\text{latent}}.
$$

It also makes cross-attention conditioning a central design pattern for image generation.

| Design Choice | Why It Matters |
| --- | --- |
| pretrained autoencoder | separates compression from generative prior |
| latent denoising | reduces compute and memory |
| U-Net denoiser | preserves spatial inductive bias |
| cross-attention | flexible conditioning interface |
| decoder | converts latent samples to pixels |
| compression factor | quality/compute bottleneck |

This note connects [[papers/architectures/ddpm|DDPM]], [[concepts/architectures/cross-attention|cross-attention]], and modern text-to-image systems.

## Evidence Pattern

The paper supports the architecture with:

| Evidence | What It Supports |
| --- | --- |
| autoencoder compression studies | useful compute/fidelity tradeoff exists |
| image generation benchmarks | latent diffusion can remain competitive with pixel models |
| inpainting and super-resolution | latent diffusion works beyond unconditional generation |
| conditioning experiments | cross-attention supports flexible inputs |
| compute comparison | latent-space denoising reduces cost |

For reading, the key ablation is not only sample quality. It is the tradeoff curve between reconstruction fidelity, latent compression, and diffusion compute.

## Practical Reading Checks

| Question | Why |
| --- | --- |
| What is the latent downsampling factor $f$? | controls compute and reconstruction quality |
| Is the autoencoder frozen during diffusion training? | separates representation from prior |
| What loss trained the autoencoder? | perceptual quality affects final samples |
| Where does cross-attention enter the U-Net? | controls conditioning capacity |
| What condition encoder is used? | text/layout/mask quality depends on it |
| What sampler and guidance are used? | sample quality may not be architecture alone |
| Are metrics pixel, perceptual, CLIP-based, or human preference? | evaluation target changes conclusions |

## Limits

- The decoder can cap final visual detail.
- Compression can remove information that diffusion cannot recover.
- The denoiser still requires iterative sampling.
- Conditioning quality depends on the condition encoder and training data.
- Cross-attention can bind prompts imperfectly.
- Compute is reduced, not eliminated.
- Benchmark gains can depend on sampler, guidance, data, and metric choice.

The concise limitation:

$$
\text{latent diffusion is efficient}
\neq
\text{free high-resolution generation}.
$$

## What To Remember

- LDM moves diffusion from pixels to a pretrained autoencoder latent.
- The core generation path is encode, denoise in latent space, decode.
- The denoising objective is DDPM-style but applied to $z$.
- Cross-attention makes conditioning flexible and reusable.
- The compression factor is an architecture bottleneck.
- Stable Diffusion is a major descendant, but the paper's durable idea is broader than one system.

## Links

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/architectures/autoencoder|Autoencoder]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/modalities/image|Image]]
- [[papers/architectures/ddpm|Denoising Diffusion Probabilistic Models]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/u-net|U-Net]]
