---
title: Deep Residual Learning for Image Recognition
aliases:
  - papers/resnet
  - papers/deep-residual-learning
tags:
  - papers
  - architectures
  - cnn
  - residual-network
---

# Deep Residual Learning for Image Recognition

> The paper introduced residual learning as a practical architecture pattern for training very deep convolutional networks.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Deep Residual Learning for Image Recognition |
| Authors | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun |
| Year | 2015 preprint; 2016 conference |
| Venue | CVPR 2016 |
| arXiv | [1512.03385](https://arxiv.org/abs/1512.03385) |
| Status | verified |

## Question

Before ResNet, simply stacking more convolutional layers often made optimization worse even when overfitting was not the only issue. The question was whether a deep network could be parameterized so that additional layers could learn refinements instead of having to relearn the full mapping.

## Main Claim

Residual blocks make very deep CNNs easier to optimize by learning a residual function around an identity path.

Narrowed claim:

$$
H(x) = x + F(x)
$$

where the block learns $F(x)$ rather than forcing the stacked layers to learn $H(x)$ directly.

This is an architecture and optimization claim, not just a computer vision benchmark claim.

## Method

The core block adds a skip connection around a small stack of convolution, normalization, and activation layers:

$$
y = F(x, W_i) + x
$$

When dimensions change, the identity path may use a projection:

$$
y = F(x, W_i) + W_s x
$$

The key idea is not that the skip path is expressive. The key idea is that the network can keep an identity-like solution available while learning residual corrections.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Residual networks optimize better than plain deep networks | comparisons between plain and residual networks at increasing depth | evidence is centered on vision benchmarks |
| Very deep CNNs can improve recognition accuracy | ImageNet and CIFAR experiments with deep residual networks | later results depend on training recipes and data |
| Residual learning generalizes beyond classification | object detection experiments using residual backbones | detection pipeline includes other components |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | image classification |
| Main datasets | ImageNet, CIFAR |
| Secondary task | object detection |
| Main comparison | plain CNNs and prior CNN backbones |
| Main metric | classification error and detection metrics |
| Not directly tested | sequence modeling, graph learning, language modeling, molecular modeling |

## Limitations

- Residual connections do not remove the need for normalization, initialization, data augmentation, or training recipe choices.
- The paper does not prove that deeper is always better; it shows a practical parameterization that makes depth more useful.
- Later residual architectures changed block ordering, bottleneck design, normalization, activation, and width-depth tradeoffs.
- Skip connections can make attribution harder because information flows through multiple paths.

## Why It Matters

ResNet made residual connections a default architectural primitive. In this wiki, it is the anchor paper for:

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/normalization|Normalization]]

## Connections

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/activation-function|Activation function]]
- [[papers/architectures/index|Architecture papers]]
