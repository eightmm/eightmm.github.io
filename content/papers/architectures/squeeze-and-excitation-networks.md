---
title: Squeeze-and-Excitation Networks
aliases:
  - papers/senet
  - papers/se-net
  - papers/squeeze-excitation
tags:
  - papers
  - architectures
  - cnn
  - vision
  - attention
---

# Squeeze-and-Excitation Networks

> The paper made channel-wise feature recalibration a reusable CNN block.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Squeeze-and-Excitation Networks |
| Authors | Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu |
| Year | 2018 |
| Venue | CVPR 2018 |
| arXiv | [1709.01507](https://arxiv.org/abs/1709.01507) |
| CVF | [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) |
| Status | verified |

## Question

Convolutions mix spatial and channel information locally, but standard CNN blocks usually treat output channels as a fixed set of feature maps. SENet asks:

$$
\text{Can a CNN adaptively reweight channels based on the current input?}
$$

The answer is a small squeeze-and-excitation block:

$$
\text{global summary}
\rightarrow
\text{channel gate}
\rightarrow
\text{feature recalibration}.
$$

## Main Claim

Squeeze-and-Excitation blocks explicitly model channel interdependencies and improve CNN representations with small additional cost.

The durable claim is:

$$
\text{spatial convolution}
+
\text{input-dependent channel gate}
\Rightarrow
\text{stronger CNN feature hierarchy}.
$$

This is a reusable architecture-block paper.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | feature map $U\in\mathbb{R}^{H\times W\times C}$ |
| Output | recalibrated feature map $\tilde{U}\in\mathbb{R}^{H\times W\times C}$ |
| Squeeze | global average pooling over spatial positions |
| Excitation | small bottleneck MLP producing channel weights |
| Recalibration | channel-wise multiplication |
| Insert point | can be added to many CNN blocks |
| Main bias | input-dependent channel importance |

## Squeeze

Given feature map:

$$
U\in\mathbb{R}^{H\times W\times C},
$$

the squeeze operation computes a channel descriptor:

$$
z_c
=
\frac{1}{H W}
\sum_{u=1}^{H}\sum_{v=1}^{W}
U_{u,v,c}.
$$

This produces:

$$
z\in\mathbb{R}^{C}.
$$

The descriptor is global over space and channel-specific.

## Excitation

The excitation network maps the channel descriptor to a gate:

$$
s
=
\sigma
\left(
W_2 \delta(W_1 z)
\right),
$$

where:

| Symbol | Meaning |
| --- | --- |
| $W_1$ | reduce channels from $C$ to $C/r$ |
| $\delta$ | nonlinearity such as ReLU |
| $W_2$ | expand channels from $C/r$ to $C$ |
| $\sigma$ | sigmoid gate |
| $r$ | reduction ratio |

The output is:

$$
s\in(0,1)^C.
$$

## Recalibration

Each channel is rescaled:

$$
\tilde{U}_{u,v,c}
=
s_c U_{u,v,c}.
$$

This lets the block emphasize or suppress channels depending on the input.

| Step | Operation | Effect |
| --- | --- | --- |
| squeeze | spatial global pooling | summarize channel response |
| excitation | bottleneck MLP | model channel dependency |
| scale | channel-wise multiplication | recalibrate features |

## Relation to Attention

SE blocks are often described as channel attention. They do not compute token-token attention like [[concepts/architectures/attention|attention]] in Transformers. Instead, they compute a channel gate:

$$
\text{SE: } U \mapsto s(U)\odot U.
$$

The similarity is input-dependent weighting. The difference is the axis:

| Mechanism | Weighted Axis |
| --- | --- |
| SE block | channels |
| spatial attention | spatial positions |
| self-attention | tokens or patches |

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet classification | SE blocks improve strong CNN backbones |
| insertion into existing networks | the block is modular rather than a full-only architecture |
| ILSVRC 2017 result | SE-based models were competitive at large scale |

## Limits

- Global average pooling discards spatial arrangement before channel gating.
- The block improves channel recalibration, but does not replace spatial modeling.
- The extra MLP is small but not zero-cost.
- Gains can depend on baseline strength, training recipe, and where the block is inserted.

## Concepts

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/computational-complexity|Computational complexity]]

## Related

- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/resnext|ResNeXt]]
- [[papers/architectures/mobilenetv2|MobileNetV2]]
- [[papers/architectures/efficientnet|EfficientNet]]
