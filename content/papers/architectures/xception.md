---
title: Xception
aliases:
  - papers/xception
  - papers/deep-learning-with-depthwise-separable-convolutions
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# Xception

> The paper reinterpreted Inception as a spectrum and pushed it to depthwise separable convolution.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Xception: Deep Learning with Depthwise Separable Convolutions |
| Author | Francois Chollet |
| Year | 2017 |
| Venue | CVPR 2017 |
| arXiv | [1610.02357](https://arxiv.org/abs/1610.02357) |
| Status | verified |

## Question

[[papers/architectures/inception|Inception]] uses parallel towers to process channels through different spatial filters. Xception asks whether this hand-designed tower structure is an intermediate point between dense convolution and a stronger factorization:

$$
\text{dense convolution}
\rightarrow
\text{Inception towers}
\rightarrow
\text{depthwise separable convolution}.
$$

The question is:

$$
\text{What happens if each channel gets its own spatial filter and channel mixing is fully separated?}
$$

## Main Claim

Xception replaces Inception modules with depthwise separable convolution and residual connections. The central architecture claim is:

$$
\text{cross-channel correlation}
\quad\text{and}\quad
\text{spatial correlation}
\quad
\text{can be modeled separately}.
$$

This turns an Inception-like multi-branch idea into a simpler repeated block.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor or image feature map |
| Output | classification logits or transferable image features |
| Core block | depthwise separable convolution with residual connection |
| Spatial operation | channel-wise spatial convolution |
| Channel operation | pointwise $1\times1$ convolution |
| Main comparison | Inception V3 under similar parameter count |
| Main bias | decouple spatial filtering from channel mixing |

## From Inception to Xception

An Inception module can be written abstractly as:

$$
y
=
\operatorname{Concat}
\left[
f_1(x), f_2(x), \dots, f_B(x)
\right],
$$

where each branch uses a different transform.

Xception interprets depthwise separable convolution as an extreme Inception module:

$$
B \approx C,
$$

where each channel has its own spatial tower.

## Depthwise Separable Block

Let:

$$
x\in\mathbb{R}^{H\times W\times C}.
$$

Depthwise spatial filtering:

$$
z_{u,v,c}
=
\sum_{\Delta u,\Delta v}
D_{\Delta u,\Delta v,c}
x_{u+\Delta u,v+\Delta v,c}.
$$

Pointwise channel mixing:

$$
y_{u,v,k}
=
\sum_{c=1}^{C}
P_{c,k}z_{u,v,c}.
$$

The block separates:

| Operation | Mixes Space? | Mixes Channels? |
| --- | --- | --- |
| depthwise convolution | yes | no |
| pointwise convolution | no | yes |
| dense convolution | yes | yes |

This separation is the key architectural assumption.

## Residual Stack

Xception is not only a single separable convolution. It uses a linear stack of separable convolution blocks with residual connections:

$$
y = x + F(x)
$$

when shapes match, or a projection shortcut when downsampling or changing channels.

The result is closer to:

$$
\text{ResNet training scaffold}
+
\text{Inception-style factorization}
+
\text{depthwise separable convolution}.
$$

## Why It Matters

Xception is useful for the architecture shelf because it gives a conceptual bridge:

| Paper | Main Design Axis |
| --- | --- |
| [Inception](/papers/architectures/inception) | manually allocated multi-branch receptive fields |
| Xception | extreme channel-wise tower factorization |
| [MobileNets](/papers/architectures/mobilenets) | efficient depthwise separable backbone for deployment |
| [MobileNetV2](/papers/architectures/mobilenetv2) | inverted residual with linear bottleneck |

The idea later appears in efficient vision backbones and encoder blocks where spatial mixing and channel mixing are separated.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet comparison | Xception can match or slightly improve Inception V3 under similar parameter count |
| large-scale classification comparison | gains are stronger in a larger training regime |
| architectural ablation framing | performance is attributed to parameter use, not simply parameter count |

## Limits

- The paper's core evidence is classification-centric.
- Depthwise separable convolution reduces arithmetic, but deployment speed depends on kernels and hardware.
- The channel/spatial independence assumption is useful, not universally optimal.
- Later efficient backbones often combine separability with residual, attention, squeeze-excitation, or search/scaling rules.

## Concepts

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/computational-complexity|Computational complexity]]

## Related

- [[papers/architectures/inception|Going Deeper with Convolutions]]
- [[papers/architectures/mobilenets|MobileNets]]
- [[papers/architectures/mobilenetv2|MobileNetV2]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
