---
title: Network In Network
aliases:
  - papers/network-in-network
  - papers/nin
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# Network In Network

> The paper replaced simple linear convolutional filters with local micro-networks and helped make $1\times1$ convolution and global average pooling standard CNN design tools.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Network In Network |
| Authors | Min Lin, Qiang Chen, Shuicheng Yan |
| Year | 2013 preprint; 2014 conference |
| Venue | ICLR 2014 |
| arXiv | [1312.4400](https://arxiv.org/abs/1312.4400) |
| OpenReview | [Network In Network](https://openreview.net/forum?id=ylE6yojDR5yqX) |
| Status | full note started |

## Question

Early convolutional networks used filters that were linear over each local patch, followed by a nonlinearity:

$$
y_{i,j,k}
=
\sigma(w_k^\top x_{i,j} + b_k).
$$

where $x_{i,j}$ is the patch around spatial location $(i,j)$.

The paper asks:

$$
\text{Can each local receptive field be processed by a small neural network instead of one linear filter?}
$$

It also asks whether the classifier head can be made more structurally tied to class feature maps rather than relying on large fully connected layers.

## Main Claim

Network In Network replaces a generalized-linear convolution filter with a small multilayer perceptron applied at each spatial location.

The local mapping becomes:

$$
z_{i,j}^{(1)}
=
\sigma(W^{(1)}x_{i,j}+b^{(1)})
$$

$$
z_{i,j}^{(2)}
=
\sigma(W^{(2)}z_{i,j}^{(1)}+b^{(2)})
$$

$$
y_{i,j}
=
\sigma(W^{(3)}z_{i,j}^{(2)}+b^{(3)}).
$$

This is the paper's mlpconv idea:

$$
y_{i,j}
=
\operatorname{MLP}_{\theta}(x_{i,j}).
$$

When implemented convolutionally, the later layers of the local MLP are $1\times1$ convolutions.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image or feature map $X\in\mathbb{R}^{H\times W\times C}$ |
| Local unit | micro-network over each receptive field |
| Practical block | spatial convolution followed by $1\times1$ convolutions |
| Token mixing | spatial locality from convolutional receptive fields |
| Channel mixing | $1\times1$ convolution mixes channels at each spatial location |
| Head | global average pooling over class-related feature maps |
| Core bias | stronger local patch abstraction with fewer dense-head parameters |

The block keeps the convolutional sharing pattern:

$$
y_{i,j}
=
f_\theta(x_{i,j}),
\qquad
\theta \text{ shared over all } (i,j).
$$

The difference is that $f_\theta$ is no longer a single affine map plus activation. It is a small network.

## From Linear Convolution To MLPConv

A conventional convolutional layer at location $(i,j)$ is:

$$
y_{i,j,k}
=
\sigma
\left(
\sum_{u,v,c}
W_{u,v,c,k}X_{i+u,j+v,c}
+
b_k
\right).
$$

This is linear in the local patch before the activation.

An mlpconv layer adds nonlinear processing inside the local patch:

$$
h_{i,j}^{(1)}
=
\sigma(W_1 * X_{i,j} + b_1)
$$

$$
h_{i,j}^{(2)}
=
\sigma(W_2 h_{i,j}^{(1)} + b_2)
$$

$$
y_{i,j}
=
\sigma(W_3 h_{i,j}^{(2)} + b_3).
$$

Here $W_2$ and $W_3$ operate channel-wise at the same spatial location. In CNN implementation terms:

$$
W_2, W_3
\quad\Longleftrightarrow\quad
1\times1\text{ convolutions}.
$$

So NiN is one of the reasons $1\times1$ convolution became a standard architecture primitive.

## What $1\times1$ Convolution Does

For a feature tensor:

$$
X\in\mathbb{R}^{H\times W\times C_{\text{in}}},
$$

a $1\times1$ convolution computes:

$$
Y_{i,j,k}
=
\sum_{c=1}^{C_{\text{in}}}
W_{c,k}X_{i,j,c}+b_k.
$$

It does not mix neighboring spatial positions. It mixes channels at each position:

$$
\mathbb{R}^{C_{\text{in}}}
\rightarrow
\mathbb{R}^{C_{\text{out}}}
\quad
\text{independently for each }(i,j).
$$

This is useful for:

| Use | Why It Matters |
| --- | --- |
| Channel mixing | learns cross-channel feature interactions |
| Dimension reduction | reduces channels before expensive spatial convolution |
| Dimension expansion | increases feature capacity after local processing |
| Nonlinear local MLP | stacks pointwise channel projections with activations |
| Bottleneck blocks | later used heavily in Inception, ResNet bottlenecks, MobileNet-style blocks |

In modern notation, many CNN blocks are:

$$
\text{spatial mixing}
\rightarrow
\text{channel mixing}
\rightarrow
\text{nonlinearity}.
$$

NiN made the channel-mixing part explicit.

## Global Average Pooling

Traditional CNNs often ended with large fully connected layers:

$$
\operatorname{flatten}(X)
\rightarrow
\operatorname{MLP}
\rightarrow
\text{class logits}.
$$

NiN instead uses global average pooling over final feature maps:

$$
\ell_k
=
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
X_{i,j,k}.
$$

where $\ell_k$ is the logit or class-related score for class $k$.

This changes the classifier head:

| Head Type | Computation | Risk |
| --- | --- | --- |
| fully connected head | flatten all spatial features and learn dense classifier | many parameters, overfitting, weak spatial correspondence |
| global average pooling | average class-related maps | fewer parameters, stronger feature-map/class correspondence |

This idea later appears in many CNN families as a default classification head.

## Relation To Inception

[[papers/architectures/inception|Inception]] uses $1\times1$ convolutions as projection and dimension-reduction modules:

$$
1\times1
\rightarrow
3\times3
$$

or:

$$
1\times1
\rightarrow
5\times5.
$$

NiN helps explain why this is sensible: $1\times1$ convolution is not a spatial filter. It is a learned channel mixer and local feature projector.

| NiN Idea | Later Inception Use |
| --- | --- |
| local micro-network | multi-branch local transformations |
| $1\times1$ channel mixing | projection before expensive convolutions |
| global average pooling | parameter-light classifier head |
| stronger local abstraction | deeper and wider CNN modules |

So NiN should be read before Inception if the goal is to understand why CNN modules started using pointwise projections.

## Relation To Modern ConvNets

Many later ConvNet blocks separate spatial mixing and channel mixing:

$$
\text{depthwise spatial convolution}
\rightarrow
1\times1\text{ pointwise convolution}.
$$

This appears in:

- [[papers/architectures/mobilenets|MobileNets]];
- [[papers/architectures/mobilenetv2|MobileNetV2]];
- [[papers/architectures/convnext|ConvNeXt]];
- inverted bottleneck and MLP-like ConvNet blocks.

NiN is not the final form of these architectures, but it is an early paper that makes the pointwise channel-mixing operation conceptually central.

## Evidence To Read Carefully

The paper reports strong results on CIFAR-10, CIFAR-100, SVHN, and MNIST for the proposed NIN structure.

For architecture reading, split the claims:

| Claim | Evidence Type | Caution |
| --- | --- | --- |
| Local micro-networks improve patch abstraction | classification results with mlpconv stacks | task scale is smaller than ImageNet-scale backbones |
| Global average pooling reduces dense-head overfitting | replacement of large fully connected layers | effectiveness depends on final feature-map design |
| $1\times1$ layers are useful channel mixers | successful stacked mlpconv architecture | later papers use $1\times1$ for additional reasons such as bottleneck efficiency |

The durable value is the design primitive, not a single benchmark result.

## Failure Modes

| Failure Mode | Mechanism | Practical Check |
| --- | --- | --- |
| Too little spatial context | $1\times1$ layers alone do not mix positions | ensure spatial convolution or attention exists |
| Channel bottleneck too narrow | projection discards information | inspect width ratios and ablations |
| Class-map assumption too strong | global average pooling assumes class evidence is spatially aggregatable | check localization and multi-object settings |
| Over-crediting $1\times1$ conv | gains may come from depth, nonlinearity, regularization, or head design | separate block ablations |

## Where It Fits

| Axis | Placement |
| --- | --- |
| Architecture family | CNN / local micro-network |
| Core primitive | $1\times1$ convolution as pointwise channel mixer |
| Classifier head | global average pooling |
| Predecessor | [AlexNet](/papers/architectures/alexnet), [VGG](/papers/architectures/vgg) |
| Successor line | [Inception](/papers/architectures/inception), [MobileNets](/papers/architectures/mobilenets), [ConvNeXt](/papers/architectures/convnext) |
| Reusable concept | [CNN](/concepts/architectures/cnn) |

## Practical Checks

When reading a CNN architecture paper, ask:

| Question | Why It Matters |
| --- | --- |
| Which operations mix space and which mix channels? | separates convolutional locality from pointwise representation mixing |
| Is $1\times1$ used for projection, bottlenecking, or expansion? | identifies compute and information-flow role |
| Is the classifier head dense or pooled? | affects parameter count and overfitting |
| Are local nonlinearities stacked inside a block? | determines block expressivity |
| Are gains from module design or training recipe? | prevents over-crediting the architecture primitive |

## Related

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/vgg|VGG]]
- [[papers/architectures/inception|Going Deeper with Convolutions]]
- [[papers/architectures/mobilenets|MobileNets]]
- [[papers/architectures/mobilenetv2|MobileNetV2]]
- [[papers/architectures/convnext|ConvNeXt]]
