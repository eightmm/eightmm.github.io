---
title: MobileNets
aliases:
  - papers/mobilenets
  - papers/mobilenetv1
  - papers/mobilenet-v1
tags:
  - papers
  - architectures
  - cnn
  - vision
  - efficient-models
---

# MobileNets

> The paper made depthwise separable convolution the default starting point for efficient mobile CNNs.

## Metadata

| Field | Value |
| --- | --- |
| Paper | MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications |
| Authors | Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam |
| Year | 2017 |
| Venue | arXiv preprint |
| arXiv | [1704.04861](https://arxiv.org/abs/1704.04861) |
| Status | verified |

## Question

Large CNNs such as [[papers/architectures/alexnet|AlexNet]], [[papers/architectures/vgg|VGG]], and [[papers/architectures/deep-residual-learning|ResNet]] are useful backbones, but many deployments care about latency, memory, and power more than leaderboard accuracy.

MobileNets asks:

$$
\text{How much CNN accuracy can be kept if dense spatial convolution is factorized?}
$$

The paper's answer is a simple backbone built mostly from depthwise separable convolutions plus two global scaling knobs.

## Main Claim

MobileNets replaces most dense convolutions with:

$$
\text{depthwise convolution}
\rightarrow
\text{pointwise convolution}.
$$

The durable claim is:

$$
\text{depthwise separable convolution}
+
\text{width multiplier}
+
\text{resolution multiplier}
\Rightarrow
\text{practical accuracy/latency control for mobile vision}.
$$

This makes the paper an architecture paper, not only an efficiency paper. It changed the default CNN block for resource-constrained models.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor $X\in\mathbb{R}^{H\times W\times C}$ |
| Output | class logits or feature maps reused for detection/classification tasks |
| Core block | depthwise separable convolution |
| Spatial mixing | per-channel $K\times K$ depthwise convolution |
| Channel mixing | $1\times1$ pointwise convolution |
| Scaling knobs | width multiplier $\alpha$, resolution multiplier $\rho$ |
| Main bias | local image structure with cheap spatial filtering |

## Dense Convolution Cost

A dense convolution with kernel size $K$, input channels $M$, output channels $N$, and feature map size $D_F\times D_F$ has approximate multiply-add cost:

$$
D_K^2 M N D_F^2.
$$

The expensive part is the product:

$$
D_K^2 \times M \times N.
$$

It simultaneously mixes spatial neighborhoods and channels.

## Depthwise Separable Convolution

MobileNet factorizes convolution into two steps.

Depthwise convolution applies one spatial filter per input channel:

$$
Z_{u,v,m}
=
\sum_{\Delta u,\Delta v}
K_{\Delta u,\Delta v,m}
X_{u+\Delta u,v+\Delta v,m}.
$$

Pointwise convolution then mixes channels:

$$
Y_{u,v,n}
=
\sum_{m=1}^{M}
P_{m,n}Z_{u,v,m}.
$$

The cost becomes:

$$
D_K^2 M D_F^2
+
M N D_F^2.
$$

The ratio against dense convolution is:

$$
\frac{D_K^2 M D_F^2 + M N D_F^2}
{D_K^2 M N D_F^2}
=
\frac{1}{N}+\frac{1}{D_K^2}.
$$

For $3\times3$ kernels and large $N$, this is much cheaper than dense convolution.

## Width and Resolution Multipliers

MobileNets expose two simple capacity controls.

The width multiplier $\alpha$ shrinks channel counts:

$$
M \rightarrow \alpha M,
\qquad
N \rightarrow \alpha N.
$$

The resolution multiplier $\rho$ shrinks input and feature map resolution:

$$
D_F \rightarrow \rho D_F.
$$

With both multipliers, depthwise separable cost is approximately:

$$
D_K^2 \alpha M (\rho D_F)^2
+
\alpha M \alpha N (\rho D_F)^2.
$$

| Knob | Changes | Main Tradeoff |
| --- | --- | --- |
| $\alpha$ | channel width | capacity vs compute |
| $\rho$ | spatial resolution | localization/detail vs compute |

These knobs make the model family deployable across different hardware budgets.

## Relation to Later CNNs

MobileNetV1 is the clean depthwise separable baseline. [[papers/architectures/mobilenetv2|MobileNetV2]] keeps depthwise separability but changes the residual block:

$$
\text{MobileNetV1: depthwise separable block}
$$

$$
\text{MobileNetV2: inverted residual + linear bottleneck}.
$$

[[papers/architectures/xception|Xception]] also uses depthwise separable convolution, but frames it as an extreme form of Inception. MobileNets frames it as an efficient deployment backbone.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet experiments | depthwise separable CNNs can keep competitive accuracy under lower computation |
| width/resolution sweeps | global multipliers provide a smooth accuracy/compute tradeoff |
| downstream tasks | the backbone transfers beyond classification |

The evidence should be read as an efficiency tradeoff claim, not a claim that MobileNetV1 dominates large CNNs under unlimited compute.

## Limits

- Latency is hardware- and kernel-dependent; fewer multiply-adds do not always mean proportionally faster wall-clock inference.
- Depthwise convolution can be memory-bandwidth-bound on some devices.
- The paper does not solve all efficient architecture design; later work improves residual structure, channel shuffle, squeeze-excitation, and neural architecture scaling.
- The width and resolution multipliers are coarse global knobs, not layer-wise allocation policies.

## Concepts

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/inference-optimization|Inference optimization]]

## Related

- [[papers/architectures/mobilenetv2|MobileNetV2]]
- [[papers/architectures/xception|Xception]]
- [[papers/architectures/efficientnet|EfficientNet]]
- [[papers/architectures/squeeze-and-excitation-networks|Squeeze-and-Excitation Networks]]
