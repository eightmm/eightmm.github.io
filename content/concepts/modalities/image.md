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

## Preprocessing Boundary

Image preprocessing is part of the model input contract:

$$
X_{\mathrm{model}}
=
\operatorname{preprocess}
(X_{\mathrm{raw}};
\text{resize},
\text{crop},
\text{color},
\text{normalization})
$$

Changing resize policy, interpolation, channel order, or normalization can change the effective distribution. For scientific images, these choices can also alter measurements, boundaries, and small structures.

## Task Coupling

The same image can support several output spaces:

$$
X
\rightarrow
\{\text{class},
\text{box set},
\text{mask},
\text{caption},
\text{embedding},
\text{generated image}\}
$$

Each task changes the valid output, loss, metric, and failure mode. For example, classification can tolerate spatial imprecision, while segmentation cannot.

## Leakage Risks

- Near-duplicate images or adjacent frames split across train and test.
- Watermarks, scanner artifacts, resolution, or acquisition device correlated with labels.
- Augmentation or crop policy that removes the evidence needed for the label.
- Metadata leakage through file names, overlays, annotations, or preprocessing masks.

## Checks

- What resolution, crop policy, and color space are used?
- Does augmentation preserve the label?
- Are near-duplicate images split across train and test?
- Is the task classification, detection, segmentation, retrieval, captioning, or generation?
- Is preprocessing fit or selected only using training data?
- Are small objects, boundaries, and low-contrast regions preserved after resizing?
- Are acquisition artifacts separated from semantic content?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
