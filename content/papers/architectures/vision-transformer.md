---
title: An Image is Worth 16x16 Words
aliases:
  - papers/vision-transformer
  - papers/vit
tags:
  - papers
  - architectures
  - transformer
  - vision
---

# An Image is Worth 16x16 Words

> The paper showed that a standard Transformer encoder can be applied to image classification by treating fixed-size image patches as tokens.

## Metadata

| Field | Value |
| --- | --- |
| Paper | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale |
| Authors | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby |
| Year | 2020 preprint; 2021 conference |
| Venue | ICLR 2021 |
| arXiv | [2010.11929](https://arxiv.org/abs/2010.11929) |
| OpenReview | [YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy) |
| Status | verified |

## Question

Before ViT, attention was often combined with convolutional vision models or used inside otherwise convolutional architectures. The question was whether an image classifier could remove convolutional inductive bias almost entirely and rely on a Transformer over image patches.

## Main Claim

With enough pretraining data, a pure Transformer encoder over patch tokens can match or exceed strong convolutional image classifiers after transfer.

Narrowed claim:

$$
X \in \mathbb{R}^{H \times W \times C}
\rightarrow
\{p_i\}_{i=1}^{N}
\rightarrow
\operatorname{TransformerEncoder}(\{e_i + \operatorname{pos}_i\})
$$

where each $p_i$ is a flattened image patch projected into a token embedding.

## Method

ViT splits the image into patches, linearly embeds each patch, adds positional embeddings, prepends a class token, and feeds the sequence to a Transformer encoder.

The patch embedding step is:

$$
z_0 =
[x_{\mathrm{class}};
x_p^1 E;
x_p^2 E;
\ldots;
x_p^N E]
+ E_{\mathrm{pos}}
$$

where $x_p^i$ is a flattened patch and $E$ is a learned projection.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Pure Transformer vision models can perform strongly | image classification transfer results after large-scale pretraining | depends heavily on pretraining data scale |
| Patch tokenization is a viable image representation | comparison against strong CNN baselines | local inductive bias is weaker than CNNs |
| Data scale changes architecture ranking | smaller-data settings favor stronger inductive bias | not all domains have large pretraining data |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | image classification |
| Input/output unit | image to class label |
| Main route | patch tokens to Transformer encoder |
| Main comparison | convolutional vision backbones and hybrid models |
| Not directly tested | dense segmentation as the core task, molecular graphs, protein structure |

## Limitations

- ViT trades convolutional locality for data-hungry global token mixing.
- The paper's headline strength depends on large-scale pretraining and transfer.
- Patch size, positional embedding, augmentation, regularization, and pretraining dataset all affect the architecture claim.
- Dense prediction and small-data vision require additional adaptations.

## Why It Matters

ViT made Transformer encoders a general vision backbone and clarified when architectural inductive bias can be replaced by data scale.

## Connections

- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
