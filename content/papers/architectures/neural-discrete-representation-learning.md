---
title: Neural Discrete Representation Learning
aliases:
  - papers/vq-vae
  - papers/neural-discrete-representation-learning
  - papers/vector-quantised-variational-autoencoder
tags:
  - papers
  - architectures
  - generative-models
  - vae
  - discrete-latent
  - vector-quantization
---

# Neural Discrete Representation Learning

> VQ-VAE replaces continuous Gaussian latents with a learned discrete codebook, making latent generative modeling look like compression plus an autoregressive prior over codes.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Discrete Representation Learning |
| Authors | Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1711.00937](https://arxiv.org/abs/1711.00937) |
| Proceedings | [NeurIPS 2017](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning) |
| Status | seed note started |

## One-Line Takeaway

VQ-VAE learns a discrete latent codebook and trains an encoder to choose nearest code vectors, allowing a decoder to reconstruct from discrete symbols and a separate autoregressive prior to generate in latent space.

## Question

Standard [[papers/architectures/auto-encoding-variational-bayes|VAE]] uses continuous latent variables:

$$
z \sim \mathcal{N}(0,I),
\qquad
x \sim p_\theta(x\mid z).
$$

With a powerful decoder, the model can ignore $z$:

$$
p_\theta(x\mid z) \approx p_\theta(x).
$$

This is posterior collapse. VQ-VAE asks:

> Can a generative model learn useful discrete latent representations while avoiding the tendency of a powerful decoder to ignore the latent code?

## Architecture Contract

| Component | Role |
| --- | --- |
| encoder | maps input to continuous latent vectors |
| codebook | stores learned discrete embedding vectors |
| vector quantization | replaces each encoder output with nearest codebook entry |
| decoder | reconstructs input from quantized code vectors |
| straight-through estimator | routes gradients from decoder to encoder |
| commitment loss | keeps encoder outputs close to selected codes |
| autoregressive prior | models the sequence or grid of discrete code indices |

The model has two paths:

$$
x
\rightarrow
z_e(x)
\rightarrow
z_q(x)
\rightarrow
\hat{x}
$$

for reconstruction, and:

$$
k_{1:N}\sim p_\psi(k_{1:N}),
\qquad
z_q = e_{k_{1:N}},
\qquad
x\sim p_\theta(x\mid z_q)
$$

for generation.

## Codebook Quantization

Let the encoder output be:

$$
z_e(x) \in \mathbb{R}^{D}.
$$

Let the codebook be:

$$
E=\{e_1,\ldots,e_K\},
\qquad
e_k\in\mathbb{R}^{D}.
$$

The discrete assignment is nearest-neighbor lookup:

$$
k^\*
=
\arg\min_{k}
\lVert z_e(x)-e_k\rVert_2.
$$

The quantized latent is:

$$
z_q(x) = e_{k^\*}.
$$

For an image, this usually happens over a spatial latent grid:

$$
z_e(x) \in \mathbb{R}^{H'\times W'\times D},
\qquad
k_{u,v}\in\{1,\ldots,K\}.
$$

So the latent representation becomes a grid of discrete symbols.

## Objective

The VQ-VAE loss has three terms:

$$
\mathcal{L}
=
-\log p_\theta(x\mid z_q(x))
+
\lVert
\operatorname{sg}[z_e(x)] - e
\rVert_2^2
+
\beta
\lVert
z_e(x)-\operatorname{sg}[e]
\rVert_2^2.
$$

where:

- $\operatorname{sg}[\cdot]$ is stop-gradient;
- the first term trains the decoder;
- the second term moves codebook embeddings toward encoder outputs;
- the third term makes the encoder commit to a code.

The straight-through estimator copies the decoder gradient through the quantization operation:

$$
\frac{\partial z_q}{\partial z_e}
\approx
I.
$$

This is not mathematically exact differentiation through nearest-neighbor lookup. It is a practical estimator that makes the architecture trainable.

## Learned Prior

After learning discrete latents, train a prior over code indices:

$$
p_\psi(k_{1:N})
=
\prod_{i=1}^{N}
p_\psi(k_i\mid k_{<i}).
$$

The prior can be a [[papers/architectures/pixel-recurrent-neural-networks|PixelCNN]] over the latent grid.

This changes the hard problem:

$$
\text{generate high-dimensional pixels directly}
$$

into:

$$
\text{generate a lower-resolution grid of discrete codes}
\rightarrow
\text{decode to pixels}.
$$

That pattern later becomes central in latent generative modeling.

## VQ-VAE vs VAE

| Axis | VAE | VQ-VAE |
| --- | --- | --- |
| latent type | continuous Gaussian | discrete codebook index |
| posterior | parametric distribution $q_\phi(z\mid x)$ | nearest-neighbor code assignment |
| regularization | KL to prior | codebook and commitment losses |
| prior | often fixed Gaussian during base training | learned prior over discrete codes |
| collapse risk | high with powerful decoders | reduced by discrete bottleneck |

VQ-VAE is still a latent-variable generative architecture, but the bottleneck is closer to learned vector quantization than a Gaussian posterior.

## Why It Matters

VQ-VAE is a bridge between autoencoders, compression, tokenization, and generative modeling.

| Contribution | Later Use |
| --- | --- |
| learned discrete visual/audio codes | tokenizer-like representations for non-text data |
| codebook bottleneck | controllable compression and discrete latent structure |
| autoregressive prior over codes | faster generation than raw-pixel autoregression |
| separation of tokenizer and prior | precursor to many latent generative pipelines |

For modern architecture reading, VQ-VAE is important because it makes a visual or audio object look like a sequence/grid of tokens.

## What To Watch

- Codebook collapse can occur when only a few codes are used.
- Reconstruction quality and generative sample quality are separate claims.
- The learned prior matters; the autoencoder alone is not a full generative model.
- Discrete codes can hide artifacts if the decoder is too strong or the bottleneck is poorly sized.
- Later latent diffusion models use continuous latent autoencoders, but the same “compress then generate in latent space” design logic is shared.

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/pixel-recurrent-neural-networks|PixelRNN / PixelCNN]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
