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

## Why It Matters

- Shows that Transformers can process non-text token sequences.
- Useful as a reference for patch, voxel, contact-map, or grid tokenization.
- Often requires more data or stronger pretraining than CNNs when locality is not hard-coded.

## Checks

- What is the patch size, and what spatial detail is lost?
- Are positional encodings absolute, relative, or learned?
- Is locality recovered through hybrid CNN stems, window attention, or data scale?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/embedding|Embedding]]
