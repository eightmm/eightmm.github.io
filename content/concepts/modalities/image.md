---
title: Image
tags:
  - modalities
  - image
  - vision
---

# Image

An image is a spatial grid of pixels or channels:

$$
X \in \mathbb{R}^{H \times W \times C}
$$

where $H$ is height, $W$ is width, and $C$ is the number of channels. Vision models must decide whether to preserve the grid structure directly or convert it into tokens.

## Common Representations

- Dense pixel tensor for [[concepts/architectures/cnn|CNN]] or U-Net-style models.
- Fixed-size patches for [[concepts/architectures/vision-transformer|Vision Transformer]] models.
- Region proposals or detected objects for object-centric pipelines.
- Feature maps from a pretrained image encoder.

Patch embedding in a Vision Transformer is:

$$
x_i = \operatorname{Flatten}(P_i)W_E + p_i
$$

where $P_i$ is image patch $i$, $W_E$ is the patch projection, and $p_i$ is a positional embedding.

## Checks

- What resolution, crop policy, and color space are used?
- Does augmentation preserve the label?
- Are near-duplicate images split across train and test?
- Is the task classification, detection, segmentation, retrieval, captioning, or generation?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
