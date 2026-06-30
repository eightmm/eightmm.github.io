---
title: ResNeXt
aliases:
  - papers/resnext
  - papers/aggregated-residual-transformations
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# ResNeXt

> The paper made cardinality a first-class CNN scaling axis beside depth and width.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Aggregated Residual Transformations for Deep Neural Networks |
| Authors | Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He |
| Year | 2017 |
| Venue | CVPR 2017 |
| arXiv | [1611.05431](https://arxiv.org/abs/1611.05431) |
| Status | verified |

## Question

[[papers/architectures/deep-residual-learning|ResNet]] showed that residual connections make very deep CNNs trainable:

$$
y = x + F(x).
$$

Inception showed that multi-branch transformations can improve representation under a compute budget. ResNeXt asks whether multi-branch design can be made simple and repeatable instead of manually tuned:

$$
\text{Can repeated homogeneous branches become a clean scaling dimension?}
$$

## Main Claim

ResNeXt introduces cardinality, the number of parallel transformations in a block:

$$
y
=
x
+
\sum_{i=1}^{C}
T_i(x),
$$

where $C$ is cardinality and each $T_i$ has the same topology.

The durable claim is:

$$
\text{increase cardinality}
\Rightarrow
\text{better accuracy/complexity tradeoff than only increasing depth or width}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image feature map |
| Output | residual block output or class logits after stacked blocks |
| Core block | aggregated residual transformation |
| New axis | cardinality $C$ |
| Implementation | grouped convolution equivalent to parallel branches |
| Main comparison | ResNet with similar complexity |
| Main bias | homogeneous multi-path feature transformations |

## Aggregated Transformations

A standard residual block computes:

$$
y = x + F(x).
$$

ResNeXt decomposes $F$ into a sum of transformations:

$$
F(x)
=
\sum_{i=1}^{C} T_i(x).
$$

Each branch has the same architecture. This differs from Inception, where branches often have hand-designed kernel sizes and channel widths.

| Model Family | Branch Design |
| --- | --- |
| Inception | heterogeneous, manually configured branches |
| ResNet | one residual transformation |
| ResNeXt | many homogeneous residual transformations |

## Cardinality

Cardinality is the number of parallel transformations:

$$
C = |\{T_1,\dots,T_C\}|.
$$

Depth counts layers. Width counts channels. Cardinality counts transformation groups.

| Scaling Axis | Changes |
| --- | --- |
| depth | number of stacked blocks |
| width | channel dimension per block |
| cardinality | number of parallel transformations |

ResNeXt argues that, under controlled complexity, increasing cardinality can be more effective than increasing only depth or width.

## Grouped Convolution View

The multi-branch sum can be implemented through grouped convolution. Suppose the hidden channels are split into $C$ groups:

$$
h = [h_1,\dots,h_C].
$$

Grouped convolution applies separate kernels:

$$
z_i = W_i * h_i,
\qquad
i=1,\dots,C.
$$

The outputs are concatenated or summed after projection:

$$
z = \operatorname{Concat}(z_1,\dots,z_C).
$$

This gives a compact implementation of the branch aggregate.

## Why It Matters

ResNeXt is a useful architecture note because it separates three concepts that are often conflated:

1. residual learning;
2. multi-branch transformations;
3. grouped convolution as an implementation mechanism.

The paper belongs between [[papers/architectures/deep-residual-learning|ResNet]], [[papers/architectures/inception|Inception]], and later efficient/modern ConvNet families.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet controlled-complexity comparisons | cardinality improves accuracy under similar complexity |
| COCO transfer experiments | the backbone improvement transfers beyond classification |
| ILSVRC 2016 system result | the architecture was competitive in large-scale vision practice |

## Limits

- Cardinality is not free; grouped convolution efficiency depends on hardware and implementation.
- The contribution is mostly architectural, not a new objective or data recipe.
- Later networks combine cardinality with attention, squeeze-excitation, depthwise convolution, or architecture search.
- The paper does not make cardinality universally superior; it shows it is a strong scaling axis under the tested regimes.

## Concepts

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/computational-complexity|Computational complexity]]

## Related

- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/inception|Going Deeper with Convolutions]]
- [[papers/architectures/squeeze-and-excitation-networks|Squeeze-and-Excitation Networks]]
- [[papers/architectures/convnext|ConvNeXt]]
