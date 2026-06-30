---
title: StyleGAN3
aliases:
  - papers/stylegan3
  - papers/alias-free-generative-adversarial-networks
  - papers/generative-models/stylegan3
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - image-generation
---

# StyleGAN3

> The paper reframed GAN generator design as a signal-processing problem: avoid aliasing so generated structure moves with the object rather than sticking to the pixel grid.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Alias-Free Generative Adversarial Networks |
| Authors | Tero Karras, Miika Aittala, Samuli Laine, Erik Harkonen, Janne Hellsten, Jaakko Lehtinen, Timo Aila |
| Year | 2021 |
| Venue | NeurIPS 2021 |
| arXiv | [2106.12423](https://arxiv.org/abs/2106.12423) |
| NeurIPS | [paper page](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html) |
| Project | [NVlabs project page](https://nvlabs.github.io/stylegan3/) |
| Code | [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) |
| Status | verified |

## Question

[[papers/architectures/stylegan2|StyleGAN2]] reduces visible artifacts with weight demodulation and generator redesign, but the synthesis process can still depend too strongly on absolute pixel coordinates.

StyleGAN3 asks:

$$
\text{Can a high-resolution GAN generator be made alias-free enough that image details transform with the represented object?}
$$

The core failure mode is coordinate locking: local details appear tied to the sampling grid rather than to the semantic surface being synthesized.

## Main Claim

StyleGAN3 treats feature maps as sampled continuous signals and redesigns generator operations to control aliasing.

The durable architecture claim is:

$$
\text{continuous signal view}
+
\text{alias-free filtering}
+
\text{controlled resampling}
\Rightarrow
\text{translation/rotation equivariant generator behavior}.
$$

This makes StyleGAN3 an architecture paper, not just a training recipe. The main object is the generator signal path.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | latent code $z$ mapped to intermediate style $w$ |
| Generator family | style-based convolutional GAN generator |
| Main change | alias-free signal processing throughout synthesis |
| Key view | feature maps are samples from continuous signals |
| Target behavior | reduce coordinate locking and improve subpixel equivariance |
| Output | generated image |
| Main comparison | StyleGAN2 quality with different internal representations |

## Coordinate Locking

A discrete feature map can be viewed as samples of an underlying continuous signal:

$$
x_s[n] = x(n\Delta),
$$

where $x(t)$ is the continuous signal, $\Delta$ is the sampling interval, and $n$ indexes grid locations.

If a generator repeatedly upsamples, applies nonlinearities, and convolves without controlling bandwidth, high-frequency content can fold into lower frequencies. That folding is aliasing.

The practical symptom is:

$$
\text{generated detail}
\not\approx
\text{object-attached detail}.
$$

Instead, hair, texture, or small image details may appear to stick to the image coordinate frame during latent interpolation or animation.

## Why Nonlinearities Matter

Even if $x$ is band-limited, a nonlinear activation can create new frequencies:

$$
y(t)=\phi(x(t)).
$$

The spectrum of $y$ may exceed the Nyquist limit of the current sampling grid. If the signal is sampled without low-pass filtering:

$$
f_{\max} > \frac{1}{2\Delta},
$$

then high-frequency components alias into lower-frequency components.

StyleGAN3's architectural move is to insert signal-processing constraints around operations that can create or move frequencies:

$$
\tilde{y}(t)=\operatorname{LPF}(\phi(x(t))).
$$

Here $\operatorname{LPF}$ is a low-pass filter chosen to keep the represented signal compatible with the sampling grid.

## Alias-Free Generator View

The generator should not smuggle absolute coordinate information through artifacts of sampling.

The idealized transformation behavior is:

$$
F(\mathcal{T}_\delta x)
\approx
\mathcal{T}_\delta F(x),
$$

where $\mathcal{T}_\delta$ translates the continuous signal by $\delta$. In words: translating the represented signal before a generator block should approximately match translating the block output.

For rotation-aware variants, the same idea extends from translation to rotations:

$$
F(\mathcal{R}_\theta x)
\approx
\mathcal{R}_\theta F(x),
$$

where $\mathcal{R}_\theta$ rotates the signal by angle $\theta$.

The paper's important point is not that a GAN becomes a perfect geometric model. It is that generator internals should commute with small image-plane transformations much better than coordinate-locked synthesis does.

## Relation to StyleGAN Lineage

| Paper | Main Architecture Contribution |
| --- | --- |
| [Progressive GAN](/papers/architectures/progressive-growing-of-gans) | grow generator and discriminator resolution during training |
| [StyleGAN](/papers/architectures/stylegan) | mapping network, per-layer styles, stochastic noise injection |
| [StyleGAN2](/papers/architectures/stylegan2) | weight modulation/demodulation and path length regularization |
| StyleGAN3 | alias-free generator signal processing and equivariance-oriented synthesis |

The lineage is useful because each step fixes a different bottleneck:

$$
\text{resolution curriculum}
\rightarrow
\text{style control}
\rightarrow
\text{artifact reduction}
\rightarrow
\text{alias-free synthesis}.
$$

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| latent interpolation analysis | StyleGAN2 details can stick to image coordinates |
| signal-processing derivation | aliasing can leak unwanted coordinate information |
| equivariance measurements | StyleGAN3 improves subpixel translation and rotation behavior |
| image quality comparisons | quality remains competitive with StyleGAN2-style baselines |

## Limits

- The result is specific to image GAN synthesis and does not by itself solve adversarial training instability.
- Alias-free design improves transformation behavior, but semantic 3D consistency is a separate problem.
- The exact implementation is still tied to generator architecture, filtering choices, dataset, and training setup.
- It is most useful when generated objects need smooth animation, interpolation, or transformation behavior.

## Concepts

- [[concepts/generative-models/gan|GAN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariance]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]

## Related

- [[papers/architectures/stylegan2|StyleGAN2]]
- [[papers/architectures/stylegan|StyleGAN]]
- [[papers/architectures/progressive-growing-of-gans|Progressive Growing of GANs]]
- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
