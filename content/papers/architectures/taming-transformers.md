---
title: Taming Transformers for High-Resolution Image Synthesis
aliases:
  - papers/taming-transformers
  - papers/vqgan
  - papers/generative-models/vqgan
tags:
  - papers
  - architectures
  - generative-models
  - transformer
  - image-generation
---

# Taming Transformers for High-Resolution Image Synthesis

> The paper introduced a VQGAN-style image tokenizer and used an autoregressive Transformer to model high-level image composition.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Taming Transformers for High-Resolution Image Synthesis |
| Authors | Patrick Esser, Robin Rombach, Bjorn Ommer |
| Year | 2020 preprint; 2021 conference |
| Venue | CVPR 2021 |
| arXiv | [2012.09841](https://arxiv.org/abs/2012.09841) |
| Project | [CompVis project page](https://compvis.github.io/taming-transformers/) |
| Code | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) |
| Status | verified |

## Question

Autoregressive Transformers can model long-range dependencies, but pixel-level image sequences are too long:

$$
H\times W\times 3
\quad
\text{tokens}
\quad
\text{is expensive for self-attention}.
$$

The paper asks:

$$
\text{Can images be compressed into a shorter learned visual vocabulary before Transformer modeling?}
$$

The answer is:

$$
\text{CNN encoder/decoder}
+
\text{vector-quantized codebook}
+
\text{adversarial/perceptual training}
+
\text{autoregressive Transformer}.
$$

## Main Claim

Taming Transformers separates high-resolution synthesis into two stages:

1. learn a discrete image codebook with a convolutional VQGAN;
2. train a Transformer prior over the code sequence.

The durable architecture claim is:

$$
\text{local perceptual image tokenizer}
+
\text{global autoregressive prior}
\Rightarrow
\text{Transformer image synthesis at high resolution}.
$$

This paper sits between [[papers/architectures/neural-discrete-representation-learning|VQ-VAE]], autoregressive Transformers, and [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]].

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input image | $x\in\mathbb{R}^{H\times W\times 3}$ |
| Encoder | maps image to latent grid $z_e(x)$ |
| Quantizer | maps latent vectors to codebook indices |
| Codebook | learned visual vocabulary $e_k$ |
| Decoder | reconstructs image from quantized codes |
| Prior | autoregressive Transformer over discrete image codes |
| Conditioning | class label, segmentation, depth, edge, or other context |
| Output | high-resolution image synthesis |

## VQGAN Tokenizer

The encoder maps an image to a latent grid:

$$
z_e(x)\in\mathbb{R}^{h\times w\times d}.
$$

Each latent vector is replaced by the nearest codebook vector:

$$
z_q(x)_{i,j}
=
e_{k^\star},
\qquad
k^\star
=
\arg\min_k
\left\|
z_e(x)_{i,j}-e_k
\right\|_2.
$$

This turns the image into discrete indices:

$$
s
=
(s_1,\dots,s_N),
\qquad
N=h w.
$$

The decoder reconstructs:

$$
\hat{x}=D(z_q(x)).
$$

The important difference from pixel tokens is:

$$
N=h w
\ll
H W.
$$

## Reconstruction and Codebook Training

The tokenizer is trained to make quantized codes reconstruct the image:

$$
\mathcal{L}_{\text{rec}}
=
\|x-\hat{x}\|.
$$

VQ-style training also uses codebook and commitment terms:

$$
\mathcal{L}_{\text{vq}}
=
\left\|
\operatorname{sg}[z_e(x)] - e
\right\|_2^2
+
\beta
\left\|
z_e(x)-\operatorname{sg}[e]
\right\|_2^2.
$$

Taming Transformers adds perceptual and adversarial pressure so that compressed codes preserve visually meaningful structure:

$$
\mathcal{L}
=
\mathcal{L}_{\text{rec}}
+
\mathcal{L}_{\text{perceptual}}
+
\mathcal{L}_{\text{GAN}}
+
\mathcal{L}_{\text{vq}}.
$$

This is why the learned codes are more useful for high-resolution synthesis than a purely pixel-level representation.

## Autoregressive Transformer Prior

After tokenization, image generation becomes sequence modeling:

$$
p(s)
=
\prod_{i=1}^{N}
p(s_i\mid s_{<i}).
$$

The Transformer learns:

$$
p_\theta(s_i\mid s_{<i}, c),
$$

where $c$ is optional conditioning.

The generated index sequence is decoded:

$$
s
\rightarrow
z_q
\rightarrow
\hat{x}.
$$

## Sliding Attention for Large Images

Even after compression, high-resolution images can produce long code sequences. The paper uses a sliding attention window so each prediction attends to a local context window rather than every previous token:

$$
p(s_i\mid s_{<i})
\approx
p(s_i\mid s_{i-L:i-1}).
$$

This is an efficiency tradeoff:

| Attention | Cost | Bias |
| --- | --- | --- |
| full causal attention | global but expensive | long-range dependency |
| sliding causal attention | cheaper | local composition window |

The codebook handles local perceptual content. The Transformer handles composition among tokens.

## Why It Matters

Taming Transformers is important because it creates a reusable pattern:

$$
\text{encode image}
\rightarrow
\text{model latent/code space}
\rightarrow
\text{decode image}.
$$

That pattern appears later in latent diffusion:

| Paper | Latent Representation | Generative Prior |
| --- | --- | --- |
| [VQ-VAE](/papers/architectures/neural-discrete-representation-learning) | discrete codebook | learned prior over codes |
| Taming Transformers | VQGAN codebook | autoregressive Transformer |
| [Latent Diffusion Models](/papers/architectures/latent-diffusion-models) | continuous/perceptual autoencoder latent | diffusion denoiser |

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| high-resolution image synthesis | codebook + Transformer can scale beyond pixel-level AR modeling |
| conditional synthesis | discrete latent prior can incorporate spatial and non-spatial conditions |
| ImageNet autoregressive results | compressed visual tokens improve AR image modeling practicality |

## Limits

- Autoregressive sampling over image codes is still sequential.
- Codebook quality bounds final image quality.
- Sliding attention introduces locality assumptions that may miss some global dependencies.
- Later latent diffusion models replace the autoregressive prior with iterative denoising for a different quality/compute tradeoff.

## Concepts

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/latent-variable-model|Latent-variable model]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/architectures/transformer|Transformer]]

## Related

- [[papers/architectures/neural-discrete-representation-learning|Neural Discrete Representation Learning]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
