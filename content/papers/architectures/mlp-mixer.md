---
title: MLP-Mixer
aliases:
  - papers/mlp-mixer
  - papers/all-mlp-architecture-for-vision
tags:
  - papers
  - architectures
  - mlp
  - vision
---

# MLP-Mixer

> The paper tested whether competitive vision backbones require convolution or attention by separating image modeling into token-mixing MLPs and channel-mixing MLPs.

## Metadata

| Field | Value |
| --- | --- |
| Paper | MLP-Mixer: An all-MLP Architecture for Vision |
| Authors | Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy |
| Year | 2021 |
| Venue | NeurIPS 2021 |
| arXiv | [2105.01601](https://arxiv.org/abs/2105.01601) |
| Proceedings | [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html) |
| Status | full note started |

## Question

CNNs build image-specific inductive bias through locality, translation sharing, and hierarchy. Vision Transformers replace most convolutional structure with global self-attention over image patches. MLP-Mixer asks a sharper architecture question:

$$
\text{Are convolution and attention necessary for strong image classification?}
$$

The paper's answer is conditional:

$$
\text{not strictly necessary, if scale, data, and regularization are strong enough.}
$$

This makes MLP-Mixer useful as an architecture control case. It removes both convolutional spatial filters and content-dependent attention, then checks how far a simpler token/channel mixing factorization can go.

## Main Claim

MLP-Mixer shows that a vision backbone can be built from MLPs alone:

$$
\text{image patches}
\rightarrow
\text{token-mixing MLP}
\rightarrow
\text{channel-mixing MLP}
\rightarrow
\text{classification head}.
$$

The durable claim is not that MLP-Mixer dominates CNNs or Transformers. The durable claim is that strong image classification does not logically require either convolution or attention:

$$
\text{convolution and attention are sufficient biases, not necessary components.}
$$

This separates two axes that are often bundled together:

| Axis | Question |
| --- | --- |
| token mixing | how do spatial positions or patches exchange information? |
| channel mixing | how does each token transform its feature vector? |

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor |
| Tokenization | split image into fixed-size non-overlapping patches |
| Patch embedding | linear projection from flattened patch to channel dimension |
| Internal representation | matrix of patches by channels |
| Token mixing | MLP across patch positions, applied per channel |
| Channel mixing | MLP across channels, applied per patch |
| Readout | global average pooling over tokens, then linear classifier |
| Natural task in paper | image classification |

For an image of height $H$, width $W$, and patch size $P$, the number of patches is:

$$
S = \frac{HW}{P^2}.
$$

After patch embedding, the Mixer input is:

$$
X \in \mathbb{R}^{S \times C},
$$

where $S$ is the number of patches and $C$ is the channel dimension.

## Mixer Block

Each Mixer block has two residual MLP sublayers:

1. token-mixing MLP;
2. channel-mixing MLP.

A compact block equation is:

$$
U = X + \left(\sigma\left(\operatorname{LN}(X)^\top W^{\mathrm{tok}}_1 + b^{\mathrm{tok}}_1\right)W^{\mathrm{tok}}_2 + b^{\mathrm{tok}}_2\right)^\top,
$$

$$
Y = U + \sigma\left(\operatorname{LN}(U)W^{\mathrm{ch}}_1 + b^{\mathrm{ch}}_1\right)W^{\mathrm{ch}}_2 + b^{\mathrm{ch}}_2.
$$

Here:

| Symbol | Meaning |
| --- | --- |
| $X, U, Y \in \mathbb{R}^{S \times C}$ | token-by-channel activations |
| $\operatorname{LN}$ | layer normalization |
| $\sigma$ | nonlinearity, usually GELU-style |
| $W^{\mathrm{tok}}_1 \in \mathbb{R}^{S \times D_S}$, $W^{\mathrm{tok}}_2 \in \mathbb{R}^{D_S \times S}$ | token-axis MLP weights |
| $W^{\mathrm{ch}}_1 \in \mathbb{R}^{C \times D_C}$, $W^{\mathrm{ch}}_2 \in \mathbb{R}^{D_C \times C}$ | channel-axis MLP weights |

The transpose is the key implementation clue. Token mixing treats each channel as a length-$S$ vector and applies an MLP over patches. Channel mixing treats each patch as a length-$C$ vector and applies an MLP over features.

## Token Mixing vs Channel Mixing

| Sublayer | Axis Mixed | Applied To | Similar Role |
| --- | --- | --- | --- |
| token-mixing MLP | patch positions | each channel column | spatial/token interaction |
| channel-mixing MLP | feature channels | each patch row | Transformer feed-forward block |

In a Transformer encoder, token mixing is content-dependent self-attention:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

In MLP-Mixer, token mixing is a learned but input-independent function over patch positions. That removes attention's pairwise content-dependent routing:

$$
\text{attention token mixing}
\neq
\text{fixed learned token-axis MLP}.
$$

This makes the model simpler, but also less adaptive to variable sequence length and pairwise image content.

## Relation to CNN, ViT, and ConvNeXt

| Model Family | Token or Spatial Mixing | Channel Mixing | Main Bias |
| --- | --- | --- | --- |
| CNN | convolution over local neighborhoods | pointwise convolution or dense channel mixing | locality and translation sharing |
| ViT | global self-attention over patches | Transformer FFN | global content-dependent token mixing |
| MLP-Mixer | token-axis MLP over patches | channel-axis MLP | learned fixed patch interaction |
| ConvNeXt | depthwise convolution | pointwise MLP-style channel mixing | modernized local ConvNet bias |

MLP-Mixer is therefore best read between [[papers/architectures/vision-transformer|Vision Transformer]] and [[papers/architectures/convnext|ConvNeXt]]. It keeps ViT-style patchification but removes attention, then makes the axis separation explicit.

## Complexity and Resolution Dependence

Let the token-mixing hidden width be $D_S$ and the channel-mixing hidden width be $D_C$.

Token mixing cost is roughly:

$$
O(C S D_S),
$$

because a token-axis MLP is applied for each channel.

Channel mixing cost is roughly:

$$
O(S C D_C),
$$

because a channel-axis MLP is applied for each token.

If the token-mixing MLP expands proportionally to $S$, the spatial mixing parameters and compute become tied to the number of patches. This is different from convolution, which shares a local kernel over spatial positions, and from attention, which can operate over variable token counts with the same projection weights.

## Evidence to Read Carefully

The main evidence is image-classification performance under large-scale pretraining or strong regularization. The paper is most convincing as a comparison against the assumption that attention or convolution must be present in every strong vision backbone.

The evidence should be read with these qualifiers:

| Qualifier | Why It Matters |
| --- | --- |
| large data | weak image-specific bias often needs more data |
| regularization | training recipe can close much of the gap |
| classification focus | dense prediction and localization are less directly tested |
| fixed patch grid | token-mixing parameters depend on the patch count |
| no content-dependent routing | interactions are learned by position, not by pairwise token content |

## What This Paper Teaches

MLP-Mixer is useful because it exposes a model-design decomposition:

$$
\text{backbone}
=
\text{representation units}
+
\text{token mixing}
+
\text{channel mixing}
+
\text{readout}.
$$

For architecture reading, this is more important than treating it as a permanent state-of-the-art model. It gives a clean vocabulary for asking whether a newer backbone improves because of:

- better token mixing;
- better channel mixing;
- stronger training recipe;
- more data;
- better scaling;
- stronger task-specific inductive bias.

## Limits

- The token-mixing MLP is tied to the number and ordering of image patches.
- It has weaker locality and translation bias than CNNs.
- It lacks the content-dependent token interactions of self-attention.
- The paper's strongest results depend on scale or modern regularization.
- It is primarily an image-classification backbone, not a complete dense prediction recipe.

## Concepts

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]

## Related

- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/convnext|ConvNeXt]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/swin-transformer|Swin Transformer]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
