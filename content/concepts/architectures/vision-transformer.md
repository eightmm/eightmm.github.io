---
title: Vision Transformer
tags:
  - architectures
  - transformer
  - vision
---

# Vision Transformer

A Vision Transformer applies Transformer blocks to image or grid patches. It replaces convolutional locality with tokenization plus attention over patches.

Patch embedding is:

$$
x_i = \operatorname{Flatten}(P_i)W_E + p_i
$$

where $P_i$ is image patch $i$, $W_E$ is the patch projection, and $p_i$ is a positional embedding.

For an image $X\in\mathbb{R}^{H\times W\times C}$ and patch size $P$, the number of tokens is:

$$
N = \frac{H}{P}\cdot\frac{W}{P}
$$

Self-attention then scales as:

$$
O(N^2 d)
$$

so smaller patches preserve more detail but increase attention cost. A class token or pooled patch representation is commonly used for image-level prediction:

$$
z = h_{\mathrm{cls}}
\quad\text{or}\quad
z = \frac{1}{N}\sum_{i=1}^{N} h_i
$$

## Inductive Bias

Compared with CNNs, ViTs encode less hard locality and more global token mixing. This can be useful with large pretraining, but weak locality can be data-hungry. Hybrid stems, window attention, hierarchical patch merging, or strong augmentation often reintroduce local bias.

## Why It Matters

- Shows that Transformers can process non-text token sequences.
- Useful as a reference for patch, voxel, contact-map, or grid tokenization.
- Often requires more data or stronger pretraining than CNNs when locality is not hard-coded.

## Checks

- What is the patch size, and what spatial detail is lost?
- Are positional encodings absolute, relative, or learned?
- Is locality recovered through hybrid CNN stems, window attention, or data scale?

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
