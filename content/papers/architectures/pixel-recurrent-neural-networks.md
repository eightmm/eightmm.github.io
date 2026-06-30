---
title: Pixel Recurrent Neural Networks
aliases:
  - papers/pixel-rnn
  - papers/pixelcnn
  - papers/pixel-recurrent-neural-networks
tags:
  - papers
  - architectures
  - generative-models
  - autoregressive-model
  - image-generation
---

# Pixel Recurrent Neural Networks

> PixelRNN and PixelCNN make image generation a tractable autoregressive density modeling problem by factorizing an image into ordered pixel conditionals.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Pixel Recurrent Neural Networks |
| Authors | Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu |
| Year | 2016 |
| Venue | ICML 2016 |
| arXiv | [1601.06759](https://arxiv.org/abs/1601.06759) |
| PMLR | [v48/oord16](https://proceedings.mlr.press/v48/oord16.html) |
| Status | seed note started |

## One-Line Takeaway

PixelRNN shows that images can be modeled with exact likelihood by predicting each pixel from previous pixels under a fixed raster ordering, while PixelCNN replaces slower recurrent dependencies with masked convolutions.

## Question

For an image:

$$
x \in \{0,\ldots,255\}^{H\times W\times C},
$$

a generative model should define:

$$
p_\theta(x).
$$

The problem is that the joint distribution over all pixels is high-dimensional. PixelRNN asks:

> Can a neural network model the exact image likelihood by decomposing the image into sequential conditional distributions?

## Autoregressive Factorization

Flatten the image into an ordered sequence:

$$
x = (x_1,\ldots,x_N),
\qquad
N = HWC.
$$

Then:

$$
p_\theta(x)
=
\prod_{i=1}^{N}
p_\theta(x_i \mid x_1,\ldots,x_{i-1}).
$$

Training minimizes negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-
\sum_{i=1}^{N}
\log
p_\theta(x_i \mid x_{<i}).
$$

This is exact under the chosen ordering. There is no adversarial discriminator, no variational lower bound, and no approximate reverse process.

## Architecture Contract

| Component | Role |
| --- | --- |
| pixel ordering | defines which pixels are visible when predicting each target pixel |
| masked dependency | prevents access to future pixels |
| recurrent or masked-convolutional stack | computes context-aware hidden state |
| output distribution | predicts discrete pixel intensity probabilities |
| teacher forcing | trains all pixel conditionals in parallel over observed images |
| ancestral sampling | generates pixels one at a time |

The model is a density estimator:

$$
x
\rightarrow
\log p_\theta(x).
$$

Sampling is slower:

$$
\hat{x}_1 \sim p_\theta(x_1),
\quad
\hat{x}_2 \sim p_\theta(x_2\mid \hat{x}_1),
\quad
\ldots
$$

## PixelRNN

PixelRNN uses recurrent layers over spatial dimensions so each pixel representation can depend on previous pixels:

$$
h_i
=
f_\theta(x_{<i}, h_{<i}),
$$

$$
p_\theta(x_i \mid x_{<i})
=
\operatorname{softmax}(W h_i).
$$

The architecture contribution is not recurrence in time, but recurrence over a 2D image ordering. This gives expressive dependencies but makes training and sampling expensive.

## PixelCNN

PixelCNN replaces explicit recurrence with masked convolutions. A masked convolution computes:

$$
h_{u,v}
=
\sum_{\Delta u,\Delta v}
M_{\Delta u,\Delta v}
W_{\Delta u,\Delta v}
x_{u+\Delta u,\ v+\Delta v},
$$

where the mask $M$ removes positions that would reveal the current or future pixel.

The causal mask enforces:

$$
h_i = f_\theta(x_{<i}),
$$

so the likelihood factorization remains valid.

The tradeoff:

| Model | Strength | Cost |
| --- | --- | --- |
| PixelRNN | strong sequential dependency modeling | slower recurrent computation |
| PixelCNN | more parallel training through masked convolution | receptive field and blind-spot issues require careful design |

## Output Distribution

For 8-bit pixels, the output can be a categorical distribution:

$$
p_\theta(x_i=k\mid x_{<i})
=
\operatorname{softmax}(a_i)_k,
\qquad
k\in\{0,\ldots,255\}.
$$

For RGB, the channel ordering can also be autoregressive:

$$
p(r,g,b)
=
p(r)\,p(g\mid r)\,p(b\mid r,g).
$$

The key is that every generated unit has a normalized conditional probability.

## Why It Matters

PixelRNN/PixelCNN is a canonical example of [[concepts/generative-models/autoregressive-model|autoregressive modeling]] outside text.

| Contribution | Why it persisted |
| --- | --- |
| exact image likelihood | lets image generative models be evaluated as density models |
| masked spatial dependency | general recipe for causal convolution over non-text data |
| discrete pixel modeling | avoids Gaussian blur from simple regression losses |
| likelihood/sample tradeoff | shows exact likelihood does not guarantee fast sampling |

It also becomes important as a prior or decoder in later latent models, including [[papers/architectures/neural-discrete-representation-learning|VQ-VAE]].

## What To Watch

- The likelihood is exact only for the chosen ordering and output discretization.
- Raster ordering is convenient but not necessarily semantically natural.
- Sampling is sequential and can be slow at high resolution.
- Good likelihood and visually pleasing samples are related but not identical.
- PixelCNN as a decoder can become powerful enough to ignore weak latents unless the latent interface is designed carefully.

## Related

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/generative-models/sampling|Sampling]]
- [[papers/architectures/wavenet|WaveNet]]
- [[papers/architectures/neural-discrete-representation-learning|VQ-VAE]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
