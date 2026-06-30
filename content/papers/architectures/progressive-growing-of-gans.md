---
title: Progressive Growing of GANs
aliases:
  - papers/progan
  - papers/progressive-gan
  - papers/progressive-growing-of-gans
  - papers/generative-models/progressive-growing-of-gans
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - image-generation
---

# Progressive Growing of GANs

> The paper made high-resolution GAN training practical by growing the generator and discriminator from coarse to fine resolutions.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Progressive Growing of GANs for Improved Quality, Stability, and Variation |
| Authors | Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen |
| Year | 2017 preprint; 2018 conference |
| Venue | ICLR 2018 |
| arXiv | [1710.10196](https://arxiv.org/abs/1710.10196) |
| OpenReview | [Hk99zCeAb](https://openreview.net/forum?id=Hk99zCeAb) |
| Code | [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans) |
| Status | verified |

## Question

[[papers/architectures/dcgan|DCGAN]] gave a practical convolutional image-GAN recipe, but directly training high-resolution GANs remains unstable. The generator and discriminator must learn global structure and fine detail at the same time.

Progressive GAN asks:

$$
\text{Can high-resolution GAN training be stabilized by learning coarse structure first?}
$$

The answer is to start at low resolution and add layers gradually.

## Main Claim

The paper grows both networks over training:

$$
4^2
\rightarrow
8^2
\rightarrow
16^2
\rightarrow
\cdots
\rightarrow
1024^2.
$$

The durable architecture/training claim is:

$$
\text{progressive generator growth}
+
\text{progressive discriminator growth}
+
\text{fade-in transitions}
\Rightarrow
\text{more stable high-resolution GAN synthesis}.
$$

This belongs in architecture papers because it changes the network topology during training.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Latent input | $z\sim p(z)$ |
| Generator | starts from low-resolution block and adds higher-resolution blocks |
| Discriminator | mirrors the current image resolution and adds matching blocks |
| Training schedule | alternate stabilization and fade-in phases |
| Output resolution | increases during training |
| Core mechanism | fade-in interpolation between old and new paths |
| Main target | high-resolution image generation |

## Progressive Growth

At resolution $r$, the generator produces:

$$
\hat{x}^{(r)}=G^{(r)}(z).
$$

Training starts with a small network:

$$
G^{(4)}: z\rightarrow 4\times4.
$$

Then new blocks are appended:

$$
G^{(8)}
=
\operatorname{ToRGB}_{8}
\circ
B_{8}
\circ
G^{(4)}.
$$

The discriminator grows symmetrically:

$$
D^{(8)}
=
D^{(4)}
\circ
B_{8}^{D}
\circ
\operatorname{FromRGB}_{8}.
$$

The model therefore learns:

| Stage | Main Burden |
| --- | --- |
| low resolution | global layout and coarse object structure |
| middle resolution | parts and medium-scale consistency |
| high resolution | texture and fine detail |

## Fade-In Transition

When adding a new resolution, the output blends old and new paths:

$$
x_{\text{out}}
=
(1-\alpha)x_{\text{old}}
+
\alpha x_{\text{new}},
\qquad
\alpha\in[0,1].
$$

As training proceeds:

$$
\alpha:0\rightarrow1.
$$

This avoids abruptly changing the generator/discriminator interface.

The same idea applies on the discriminator side, where real and generated images at the new resolution are blended into the old path during transition.

## Why It Helps

Direct high-resolution GAN training couples too many tasks:

$$
\text{layout}
+
\text{shape}
+
\text{texture}
+
\text{local detail}
\quad
\text{all at once}.
$$

Progressive growing separates them over training time:

$$
\text{coarse}
\rightarrow
\text{medium}
\rightarrow
\text{fine}.
$$

This is not only a curriculum over data resolution. It is a curriculum over model capacity and topology.

## Practical Architecture Details

The paper also uses implementation choices that became associated with the Progressive GAN/StyleGAN lineage.

| Detail | Role |
| --- | --- |
| minibatch standard deviation | encourage sample variation |
| equalized learning rate | normalize dynamic range of weights during training |
| pixelwise feature normalization | control feature magnitude in generator |
| fade-in layers | avoid sudden topology changes |
| progressive dataset resolution | match model resolution during training |

These details matter because GAN training stability depends on the interaction between generator and discriminator.

## Relation to StyleGAN

[[papers/architectures/stylegan|StyleGAN]] inherits the high-resolution synthesis lineage but changes generator control:

| Paper | Main Idea |
| --- | --- |
| Progressive GAN | grow networks from low to high resolution |
| StyleGAN | style-based per-layer modulation and stochastic detail control |

Progressive GAN is the training-topology milestone before style-based generator control.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| high-resolution face synthesis | progressive growing can reach $1024^2$ image generation |
| stability observations | growing topology eases adversarial training |
| variation metric | evaluates both quality and diversity |
| CIFAR/ImageNet/CelebA-HQ experiments | method generalizes beyond one visual dataset |

## Limits

- Progressive growing is a training procedure, not a universal generator block.
- Fade-in schedules add implementation complexity.
- Later StyleGAN variants moved away from progressive growing in some settings.
- High visual quality can still depend heavily on dataset curation and compute.

## Concepts

- [[concepts/generative-models/gan|GAN]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]

## Related

- [[papers/architectures/dcgan|DCGAN]]
- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
- [[papers/architectures/stylegan|StyleGAN]]
- [[papers/architectures/self-attention-gans|Self-Attention GAN]]
