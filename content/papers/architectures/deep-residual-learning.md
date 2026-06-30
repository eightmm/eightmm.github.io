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

The paper is not only about making networks larger. It identifies the degradation problem: beyond some depth, a plain deep network can have higher training error than a shallower network, even though the deeper model should in principle be able to represent the shallower solution.

That observation suggests an optimization and parameterization problem:

$$
\text{more layers}
\not\Rightarrow
\text{easier optimization}
$$

The residual hypothesis is that it is easier to learn a correction to an identity mapping than to learn the full mapping from scratch.

## Main Claim

Residual blocks make very deep CNNs easier to optimize by learning a residual function around an identity path.

Narrowed claim:

$$
H(x)
=
x + F(x)
$$

where the block learns $F(x)$ rather than forcing the stacked layers to learn $H(x)$ directly.

This is an architecture and optimization claim, not just a computer vision benchmark claim.

Equivalently, if the desired mapping is $H(x)$, the residual branch learns:

$$
F(x)
=
H(x) - x
$$

If the best extra layers are unnecessary, the residual branch can move toward zero and the block can approximate identity:

$$
F(x) \approx 0
\quad\Rightarrow\quad
H(x) \approx x
$$

This identity fallback is the core design idea.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor or intermediate feature map |
| Output | feature map with same or changed spatial/channel dimensions |
| Main unit | residual block |
| Token/grid assumption | 2D image grid with convolutional locality |
| Shortcut path | identity or projection |
| Residual path | stack of convolution, normalization, activation layers |
| Main promise | deeper CNNs become easier to optimize |

For a feature map:

$$
x \in \mathbb{R}^{H \times W \times C}
$$

a same-shape residual block returns:

$$
y
=
x + F(x; W)
$$

When spatial size or channel count changes:

$$
y
=
W_s x + F(x; W)
$$

where $W_s$ is usually a projection shortcut.

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

## Basic Block

For shallower ImageNet ResNets and CIFAR-style networks, the basic residual block uses two $3 \times 3$ convolutions:

$$
F(x)
=
W_2 * \sigma(\operatorname{BN}(W_1 * x))
$$

and then:

$$
y
=
\sigma(\operatorname{BN}(F(x)) + x)
$$

where $*$ denotes convolution, $\operatorname{BN}$ is batch normalization, and $\sigma$ is usually ReLU in the original architecture.

The important point is that the residual branch does the nonlinear transformation, while the shortcut preserves a direct signal path.

## Bottleneck Block

For deeper ImageNet models, the paper uses a bottleneck design:

$$
1 \times 1
\rightarrow
3 \times 3
\rightarrow
1 \times 1
$$

The first $1 \times 1$ convolution reduces channel dimension, the $3 \times 3$ convolution performs spatial processing, and the final $1 \times 1$ convolution restores channel dimension.

| Layer | Role |
| --- | --- |
| $1 \times 1$ reduce | lower channel cost before spatial convolution |
| $3 \times 3$ conv | local spatial feature extraction |
| $1 \times 1$ expand | restore output channel width |
| shortcut | preserve direct signal path |

The bottleneck block makes very deep networks computationally practical.

## Shortcut Types

The shortcut path is not always identical.

| Shortcut | Use | Tradeoff |
| --- | --- | --- |
| identity | same shape | no extra parameters |
| zero-padding shortcut | channel increase without projection | cheap, less expressive |
| projection shortcut | shape/channel mismatch or stronger shortcut | adds parameters and compute |

When feature map size changes, a projection is needed for addition to be well-defined:

$$
\operatorname{shape}(F(x))
=
\operatorname{shape}(W_s x)
$$

This is an architecture contract, not an implementation detail. Residual addition requires shape alignment.

## Why Residuals Help Optimization

Residual networks improve gradient flow because the shortcut creates direct paths through the network.

For one block:

$$
y
=
x + F(x)
$$

The local derivative is:

$$
\frac{\partial y}{\partial x}
=
I
+
\frac{\partial F}{\partial x}
$$

The identity term gives the gradient a direct route, even if the residual branch is poorly conditioned.

Across many blocks:

$$
x_L
=
x_0
+
\sum_{l=0}^{L-1}
F_l(x_l)
$$

This additive view makes depth behave like a sequence of refinements rather than a completely new transformation at every layer.

## Relation to Plain CNNs

| Architecture | Main Computation | Failure Mode Addressed |
| --- | --- | --- |
| VGG-style plain CNN | stack convolutions directly | deeper stacks can become harder to optimize |
| Inception-style CNN | multi-branch compute-efficient modules | width and receptive-field diversity |
| ResNet | residual blocks with shortcut paths | degradation and gradient flow |

ResNet does not remove convolutional inductive bias. It changes how convolutional layers are composed.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Residual networks optimize better than plain deep networks | comparisons between plain and residual networks at increasing depth | evidence is centered on vision benchmarks |
| Very deep CNNs can improve recognition accuracy | ImageNet and CIFAR experiments with deep residual networks | later results depend on training recipes and data |
| Residual learning generalizes beyond classification | object detection experiments using residual backbones | detection pipeline includes other components |

## Ablation Reading

The paper's strongest evidence is the plain-vs-residual comparison at similar depth.

| Ablation Axis | What it tests | Reading |
| --- | --- | --- |
| plain vs residual network | whether shortcuts improve optimization | the core evidence for residual learning |
| increasing depth | whether very deep networks remain trainable | residual parameterization makes depth useful |
| shortcut option | identity vs projection behavior | projection can help shape matching and capacity |
| CIFAR and ImageNet settings | robustness across dataset scales | still vision-centered evidence |
| detection transfer | backbone reuse beyond classification | downstream system includes other design choices |

The key reading is that the residual network does not merely reduce test error; it also reduces training error compared with plain deep networks. That supports an optimization claim, not only a regularization claim.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | image classification |
| Main datasets | ImageNet, CIFAR |
| Secondary task | object detection |
| Main comparison | plain CNNs and prior CNN backbones |
| Main metric | classification error and detection metrics |
| Not directly tested | sequence modeling, graph learning, language modeling, molecular modeling |

## What Later Models Kept and Changed

| Kept | Changed Later |
| --- | --- |
| skip/residual paths | pre-activation residual blocks |
| additive identity route | gated, scaled, or normalized residual variants |
| bottleneck block idea | wider bottlenecks, grouped convolution, depthwise convolution |
| deep backbone reuse | modern detection/segmentation/self-supervised backbones |
| residual composition | Transformers and MLP-style models adopted residual blocks |

The residual connection became architecture-neutral. Transformers, diffusion U-Nets, graph networks, and many protein/structure models now use residual composition even when the original CNN setting is absent.

## Implementation Notes

- Addition requires shape alignment; channel or stride changes need a projection or other matching rule.
- BatchNorm mode matters: training/eval mismatch can change residual block behavior.
- Initialization affects whether residual branches start near identity or perturb strongly.
- Pre-activation ResNets changed the ordering of BN/ReLU/conv and often improve optimization.
- Residual connections help optimization but do not guarantee numerical stability at arbitrary depth.
- In transfer learning, early convolutional locality and later semantic features behave differently; freezing all layers may not be ideal.

## Limitations

- Residual connections do not remove the need for normalization, initialization, data augmentation, or training recipe choices.
- The paper does not prove that deeper is always better; it shows a practical parameterization that makes depth more useful.
- Later residual architectures changed block ordering, bottleneck design, normalization, activation, and width-depth tradeoffs.
- Skip connections can make attribution harder because information flows through multiple paths.
- The evidence is mostly image classification and detection, so claims about other modalities require separate support.
- Residual paths can hide weak residual branches; a deep model may behave like a shallower model if residual functions contribute little.
- Compute, memory, and latency still grow with depth even when optimization improves.

## Why It Matters

ResNet made residual connections a default architectural primitive. In this wiki, it is the anchor paper for:

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/normalization|Normalization]]

The general reusable pattern is:

$$
\text{state}
\rightarrow
\text{state}
+
\text{learned correction}
$$

This pattern appears in:

| Area | Residual-like Role |
| --- | --- |
| CNNs | train deeper visual backbones |
| Transformers | stabilize repeated attention/FFN blocks |
| diffusion U-Nets | compose denoising features across depth |
| neural ODEs | view depth as incremental dynamics |
| graph networks | avoid oversmoothing and preserve node states |

## Connections

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/weight-initialization|Weight initialization]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[papers/architectures/vgg|VGG]]
- [[papers/architectures/inception|Inception]]
- [[papers/architectures/densenet|DenseNet]]
- [[papers/architectures/neural-ode|Neural ODE]]
- [[papers/architectures/index|Architecture papers]]
