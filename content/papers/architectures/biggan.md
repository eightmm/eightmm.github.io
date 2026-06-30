---
title: BigGAN
aliases:
  - papers/biggan
  - papers/large-scale-gan-training
  - papers/generative-models/biggan
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - scaling
---

# BigGAN

> The paper showed that class-conditional GANs can improve dramatically when architecture, batch size, model size, and sampling control are scaled together.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Large Scale GAN Training for High Fidelity Natural Image Synthesis |
| Authors | Andrew Brock, Jeff Donahue, Karen Simonyan |
| Year | 2018 preprint; 2019 conference |
| Venue | ICLR 2019 |
| arXiv | [1809.11096](https://arxiv.org/abs/1809.11096) |
| OpenReview | [B1xsqj09Fm](https://openreview.net/forum?id=B1xsqj09Fm) |
| Code | [ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) |
| Status | verified |

## Question

[[papers/architectures/self-attention-gans|SAGAN]] showed that attention helps class-conditional image GANs. BigGAN asks a scaling question:

$$
\text{What happens if class-conditional GANs are scaled aggressively?}
$$

The answer is not only "make it bigger." BigGAN combines model scale, batch scale, conditional architecture, regularization, and sampling control.

## Main Claim

BigGAN improves high-fidelity class-conditional image synthesis through:

1. large generator and discriminator capacity;
2. large batch training;
3. class-conditional batch normalization;
4. residual GAN blocks;
5. shared class embeddings and hierarchical latent injection;
6. orthogonal regularization and truncation sampling.

The compact claim:

$$
\text{scaled conditional GAN}
+
\text{residual architecture}
+
\text{class-conditional normalization}
+
\text{truncation control}
\Rightarrow
\text{high-fidelity ImageNet synthesis}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input noise | latent vector $z\sim p(z)$ |
| Condition | class label $y$ |
| Generator | residual upsampling network with class-conditional normalization |
| Discriminator | residual downsampling network with class conditioning |
| Latent route | hierarchical latent chunks injected into generator blocks |
| Sampling control | truncation trick changes fidelity/diversity tradeoff |
| Main target | class-conditional ImageNet generation |
| Main risk | scale-specific GAN instability |

## Class-Conditional Generator

The generator samples:

$$
z\sim \mathcal{N}(0,I),
\qquad
y\in\{1,\dots,K\}.
$$

It produces:

$$
\hat{x}=G_\theta(z,y).
$$

Class information enters through conditional batch normalization:

$$
\operatorname{CBN}(h,y)
=
\gamma(y)
\frac{h-\mu(h)}{\sigma(h)}
+
\beta(y).
$$

This makes each class modulate the feature distribution:

$$
y
\rightarrow
(\gamma_y,\beta_y)
\rightarrow
\text{generator block behavior}.
$$

## Hierarchical Latent Injection

BigGAN splits latent $z$ into chunks:

$$
z=[z_0,z_1,\dots,z_L].
$$

Different chunks are provided to different generator blocks:

$$
h_{\ell+1}
=
B_\ell(h_\ell,z_\ell,y).
$$

This gives the generator multiple noise/control inputs across resolution levels rather than a single initial latent projection.

Conceptually:

| Route | Role |
| --- | --- |
| class label $y$ | semantic class control |
| latent chunk $z_\ell$ | stochastic variation at block $\ell$ |
| residual block | resolution-specific synthesis |

## Residual GAN Blocks

BigGAN uses residual upsampling/downsampling blocks:

$$
h_{\ell+1}
=
\operatorname{Skip}(h_\ell)
+
F_\ell(h_\ell,y,z_\ell).
$$

For the generator, blocks upsample:

$$
H\times W
\rightarrow
2H\times2W.
$$

For the discriminator, blocks downsample:

$$
H\times W
\rightarrow
\frac{H}{2}\times\frac{W}{2}.
$$

This ties BigGAN to the ResNet lineage while staying within adversarial image generation.

## Truncation Trick

At sampling time, BigGAN can restrict latent samples to a smaller region:

$$
z_i
\sim
\operatorname{TruncatedNormal}(-\tau,\tau).
$$

Smaller truncation threshold $\tau$ tends to improve fidelity but reduce diversity:

| $\tau$ | Expected Effect |
| --- | --- |
| lower | cleaner, less diverse samples |
| higher | more diverse, more failure risk |

The paper connects this to orthogonal regularization so the generator remains amenable to truncation.

## Scaling View

BigGAN is important because it treats GAN quality as a scale-sensitive system:

$$
\text{quality}
=
f(
\text{model size},
\text{batch size},
\text{conditioning},
\text{regularization},
\text{sampling}
).
$$

It belongs beside [[concepts/systems/scaling-claim-contract|scaling claim contract]] because the result is not explained by a single block.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet class-conditional metrics | large-scale conditional GANs can sharply improve IS/FID |
| truncation sweeps | sampling distribution controls fidelity/diversity |
| ablations | architecture and regularization choices matter at scale |
| instability analysis | larger GANs expose scale-specific failure modes |

## Limits

- BigGAN is compute-heavy and not a small-recipe baseline.
- Truncation can trade diversity for sample quality.
- Class-conditional ImageNet results do not automatically transfer to arbitrary domains.
- The paper is partly an architecture note and partly a scaling/optimization note.

## Concepts

- [[concepts/generative-models/gan|GAN]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]

## Related

- [[papers/architectures/self-attention-gans|Self-Attention GAN]]
- [[papers/architectures/dcgan|DCGAN]]
- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
- [[papers/architectures/stylegan|StyleGAN]]
