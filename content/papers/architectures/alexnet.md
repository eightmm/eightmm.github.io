---
title: ImageNet Classification with Deep Convolutional Neural Networks
aliases:
  - papers/alexnet
  - papers/imagenet-classification-with-deep-convolutional-neural-networks
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# ImageNet Classification with Deep Convolutional Neural Networks

> The paper showed that a large deep CNN trained on ImageNet could dramatically outperform prior computer-vision systems.

## Metadata

| Field | Value |
| --- | --- |
| Paper | ImageNet Classification with Deep Convolutional Neural Networks |
| Authors | Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton |
| Year | 2012 |
| Venue | NeurIPS 2012 |
| NeurIPS | [Proceedings page](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) |
| Status | verified |

## Question

Before AlexNet, large-scale image recognition still relied heavily on hand-engineered features and shallower models. The question was whether a large convolutional neural network, trained end-to-end on ImageNet with GPU acceleration, could dominate the benchmark.

## Main Claim

A deep convolutional network can learn visual representations directly from large labeled image data and achieve a large performance jump on ImageNet classification.

Narrowed claim:

$$
\hat{y}
=
f_\theta(X)
$$

where $f_\theta$ is a multi-layer CNN trained end-to-end from image pixels to class probabilities.

## Method

AlexNet combines several architectural and training choices:

| Component | Role |
| --- | --- |
| convolutional layers | learn local visual filters with weight sharing |
| max pooling | reduce spatial resolution and add local invariance |
| ReLU | improve optimization compared with saturating activations |
| dropout | regularize large fully connected layers |
| data augmentation | improve generalization |
| GPU training | make large CNN training practical |

The core convolutional inductive bias is:

$$
z_{i,j,k}
=
\sum_{u,v,c}
W_{u,v,c,k} X_{i+u,j+v,c}
$$

where the same filter $W$ is reused across spatial positions.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Deep CNNs outperform prior ImageNet systems | top-1 and top-5 ImageNet error improvements | result combines architecture, data scale, and GPU training |
| ReLU accelerates deep CNN training | empirical comparison against tanh-style activations | later activation functions and normalization changed the recipe |
| Dropout and augmentation reduce overfitting | training setup and benchmark performance | not isolated from all other design choices |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | image classification |
| Dataset | ImageNet LSVRC |
| Input/output unit | image to class label |
| Architecture family | CNN |
| Main metric | top-1 and top-5 classification error |
| Not directly tested | detection, segmentation, sequence modeling, graph learning |

## Limitations

- The paper is a system-level milestone as much as an architecture paper: data, compute, implementation, augmentation, and regularization all matter.
- The architecture predates modern normalization, residual connections, and efficient CNN design.
- ImageNet classification is not a complete proxy for all visual understanding.
- Later CNNs changed depth, skip connections, normalization, factorized convolutions, and training recipes.

## Why It Matters

AlexNet is the architecture paper that made deep CNNs the default starting point for vision backbones and set the stage for [[papers/architectures/deep-residual-learning|Deep Residual Learning]].

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/dropout|Dropout]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/index|Architecture papers]]
