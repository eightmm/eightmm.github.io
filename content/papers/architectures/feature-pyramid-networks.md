---
title: Feature Pyramid Networks
aliases:
  - papers/feature-pyramid-networks
  - papers/fpn
  - papers/feature-pyramid-networks-for-object-detection
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - segmentation
  - cnn
  - multi-scale
---

# Feature Pyramid Networks

> The paper turns the natural multi-scale hierarchy of a CNN into a reusable feature pyramid with top-down pathways and lateral connections.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Feature Pyramid Networks for Object Detection |
| Authors | Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie |
| Year | 2017 |
| Venue | CVPR 2017 |
| arXiv | [1612.03144](https://arxiv.org/abs/1612.03144) |
| Status | seed note started |

## One-Line Takeaway

FPN builds semantically strong feature maps at multiple spatial resolutions, so detectors can handle small and large objects without recomputing an image pyramid.

## Question

Object detection must handle scale:

$$
\text{small objects}
\leftrightarrow
\text{high-resolution features},
$$

$$
\text{large objects}
\leftrightarrow
\text{low-resolution semantic features}.
$$

Deep CNNs already form a pyramid:

$$
C_2, C_3, C_4, C_5,
$$

where shallow maps have high spatial resolution and deep maps have stronger semantics. The problem is that these properties are split across levels.

FPN asks:

> Can we make every pyramid level semantically useful while keeping the cost close to a normal backbone?

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| bottom-up pathway | image | backbone feature maps $C_l$ | standard CNN hierarchy |
| top-down pathway | deep feature maps | upsampled semantic maps | propagate high-level semantics upward |
| lateral connections | matching $C_l$ maps | fused maps | inject high-resolution spatial detail |
| output pyramid | fused maps $P_l$ | multi-scale detection features | run heads at several resolutions |

The core architecture:

$$
\{C_2,C_3,C_4,C_5\}
\rightarrow
\{P_2,P_3,P_4,P_5\}.
$$

## Top-Down Pathway

Let $C_l$ be the backbone feature map at level $l$. FPN constructs output pyramid levels recursively:

$$
P_l
=
\operatorname{Conv}_{3\times3}
\left(
\operatorname{Conv}_{1\times1}(C_l)
+
\operatorname{Up}(P_{l+1})
\right).
$$

where:

- $\operatorname{Conv}_{1\times1}$ aligns channel dimensions;
- $\operatorname{Up}$ upsamples the coarser feature map by a factor of 2;
- $\operatorname{Conv}_{3\times3}$ reduces aliasing after fusion.

This combines:

$$
\text{semantic depth from } P_{l+1}
+
\text{spatial detail from } C_l.
$$

## Why It Matters

FPN is an architecture block, not a detector by itself. It can plug into region proposal networks, two-stage detectors, segmentation heads, and later dense prediction systems.

The reusable pattern:

$$
\text{backbone}
\rightarrow
\text{multi-scale feature neck}
\rightarrow
\text{task head}.
$$

This `neck` concept became standard in many detection and segmentation systems.

## Relation To R-CNN Family

[[papers/architectures/faster-r-cnn|Faster R-CNN]] can use FPN features for proposals and detection heads:

$$
I
\rightarrow
\{P_l\}
\rightarrow
\operatorname{RPN}
\rightarrow
\operatorname{RoIHead}.
$$

[[papers/architectures/mask-r-cnn|Mask R-CNN]] can use the same pyramid for instance segmentation:

$$
I
\rightarrow
\{P_l\}
\rightarrow
\operatorname{RoIAlign}
\rightarrow
\{\text{class},\text{box},\text{mask}\}.
$$

The important architectural point is that scale handling moves from external image pyramids into the network.

## Comparison

| Approach | Multi-Scale Handling | Cost Shape |
| --- | --- | --- |
| image pyramid | run backbone at several image scales | expensive |
| single final feature map | use deepest map only | weak for small objects |
| FPN | fuse CNN pyramid levels | marginal extra cost over backbone |

## What To Watch

- FPN changes the feature extractor, not the whole detection objective.
- Gains may interact with anchor scale assignment and detector head design.
- The pyramid level chosen for an object affects small-object recall.
- Stronger backbones and augmentation can obscure the isolated value of the feature pyramid.
- Later detector families often hide FPN-like ideas under the name `neck`.

## Related

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/retinanet|RetinaNet]]
- [[papers/architectures/ssd|SSD]]
- [[papers/architectures/faster-r-cnn|Faster R-CNN]]
- [[papers/architectures/mask-r-cnn|Mask R-CNN]]
- [[papers/architectures/detr|DETR]]
