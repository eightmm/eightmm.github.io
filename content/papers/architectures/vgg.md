---
title: Very Deep Convolutional Networks for Large-Scale Image Recognition
aliases:
  - papers/vgg
  - papers/very-deep-convolutional-networks
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# Very Deep Convolutional Networks for Large-Scale Image Recognition

> The paper showed that depth with small convolution filters is a powerful and simple design rule for CNN backbones.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Very Deep Convolutional Networks for Large-Scale Image Recognition |
| Authors | Karen Simonyan, Andrew Zisserman |
| Year | 2015 |
| Venue | ICLR 2015 |
| arXiv | [1409.1556](https://arxiv.org/abs/1409.1556) |
| Author page | [Oxford VGG publication page](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/) |
| Status | verified |

## Question

After AlexNet, it was unclear how much improvement came from depth, filter size, or training recipe. The paper asked whether very deep CNNs with a simple repeated $3 \times 3$ convolution design could improve large-scale image recognition.

## Main Claim

CNN accuracy improves when depth is increased using small convolution filters and a uniform architecture pattern.

Stacking two $3 \times 3$ convolutions gives a $5 \times 5$ effective receptive field with more nonlinearities:

$$
3 \times 3
\rightarrow
3 \times 3
\quad
\approx
\quad
5 \times 5 \text{ receptive field}
$$

Three stacked $3 \times 3$ convolutions approximate a $7 \times 7$ receptive field.

## Method

| Component | Role |
| --- | --- |
| repeated $3 \times 3$ convolution | simple local feature extractor |
| depth increase | expands hierarchical representation capacity |
| max pooling | reduces spatial resolution between stages |
| fully connected classifier | maps final features to ImageNet classes |
| uniform design | makes depth the main variable under study |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Deeper CNNs improve ImageNet performance | comparison of 11 to 19 weight-layer networks | compute and parameter count also increase |
| Small filters are effective | strong results with repeated $3 \times 3$ blocks | later networks improved efficiency with residuals and bottlenecks |
| Learned features transfer | evaluation on other recognition datasets | transfer setup predates modern foundation-model evaluation |

## Limitations

- VGG is parameter-heavy and inefficient compared with later CNN backbones.
- Very deep plain CNNs become hard to optimize without residual connections.
- The design does not use normalization as a central component.
- It is a vision-specific local-grid architecture.

## Why It Matters

VGG is the cleanest classic reference for depth, small filters, and simple CNN backbone design before residual networks became dominant.

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/activation-function|Activation function]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/index|Architecture papers]]
