---
title: SSD
aliases:
  - papers/ssd
  - papers/single-shot-multibox-detector
  - papers/ssd-single-shot-multibox-detector
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - cnn
  - one-stage-detector
---

# SSD

> The paper makes object detection a single-shot prediction problem over default boxes placed on multiple feature maps.

## Metadata

| Field | Value |
| --- | --- |
| Paper | SSD: Single Shot MultiBox Detector |
| Authors | Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg |
| Year | 2016 |
| Venue | ECCV 2016 |
| arXiv | [1512.02325](https://arxiv.org/abs/1512.02325) |
| Status | seed note started |

## One-Line Takeaway

SSD removes proposal generation and predicts class scores plus box offsets for many default boxes across several feature-map resolutions.

## Question

[[papers/architectures/yolo|YOLO]] showed that object detection can be done in one forward pass, but early one-stage detectors struggled with localization and object scale.

SSD asks:

> Can a single-shot detector improve scale and aspect-ratio coverage by predicting from multiple feature maps and multiple default boxes?

The interface is:

$$
I
\rightarrow
\{F_l\}_{l=1}^{L}
\rightarrow
\{(\hat{p}_{i,a},\hat{t}_{i,a})\}.
$$

where:

- $F_l$ is a feature map at level $l$;
- $i$ is a spatial location;
- $a$ is a default box index;
- $\hat{p}$ is a class distribution;
- $\hat{t}$ is a box offset.

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| base CNN | image | feature hierarchy | visual representation |
| extra feature layers | deep features | lower-resolution maps | larger receptive fields |
| default boxes | feature locations | anchor-like priors | cover scales/aspect ratios |
| prediction heads | each feature map | class logits and box offsets | dense one-stage detection |
| NMS | candidate detections | final detections | remove duplicate boxes |

The core architecture:

$$
\text{image}
\rightarrow
\text{multi-scale feature maps}
\rightarrow
\text{default-box predictions}.
$$

## Default Boxes

At each feature-map cell, SSD defines several default boxes:

$$
D_{l,i}
=
\{d_{l,i,1},\ldots,d_{l,i,A_l}\}.
$$

Each default box has a scale and aspect ratio. The model predicts offsets:

$$
\hat{t}_{l,i,a}
=
(\Delta x,\Delta y,\Delta w,\Delta h),
$$

and class logits:

$$
\hat{p}_{l,i,a}
\in
\mathbb{R}^{C+1}.
$$

The final box is:

$$
\hat{b}_{l,i,a}
=
\operatorname{decode}(d_{l,i,a}, \hat{t}_{l,i,a}).
$$

The background class accounts for default boxes that do not match objects.

## Multi-Scale Prediction

SSD predicts from several feature maps:

$$
\{F_1,F_2,\ldots,F_L\}.
$$

Higher-resolution maps help smaller objects:

$$
F_{\text{early}}
\rightarrow
\text{small-object candidates}.
$$

Lower-resolution maps help larger objects:

$$
F_{\text{late}}
\rightarrow
\text{large-object candidates}.
$$

This is a key difference from a detector that predicts only from one final feature grid.

## Training Objective

SSD matches ground-truth boxes to default boxes and trains classification plus localization:

$$
\mathcal{L}
=
\frac{1}{N}
\left(
\mathcal{L}_{\mathrm{conf}}
+
\alpha
\mathcal{L}_{\mathrm{loc}}
\right).
$$

The localization term is usually Smooth L1 over matched boxes:

$$
\mathcal{L}_{\mathrm{loc}}
=
\sum_{(i,j)\in \mathcal{M}}
\operatorname{SmoothL1}
\left(
t_i - t_j^\*
\right),
$$

where $\mathcal{M}$ is the matching between default boxes and ground-truth boxes.

## Why It Matters

SSD is important because it makes the one-stage detector design more general than the original YOLO grid:

$$
\text{single-shot}
+
\text{default boxes}
+
\text{multi-feature-map prediction}.
$$

This pattern directly anticipates later dense detectors and mobile detection heads.

## Comparison

| Axis | [YOLO](/papers/architectures/yolo) | SSD |
| --- | --- | --- |
| Prediction locations | grid cells | multiple feature maps |
| Box priors | grid responsibility | default boxes per location |
| Scale handling | limited in original YOLO | explicit multi-level prediction |
| Detector type | one-stage | one-stage |
| Main tradeoff | speed and simplicity | more anchors/default boxes to tune |

## What To Watch

- Default-box scale/aspect design strongly affects recall.
- Hard negative mining is part of the training recipe, not only the architecture.
- Multi-scale feature prediction is not the same as the later FPN top-down fusion.
- NMS remains part of inference.
- Later SSD variants differ through backbones, feature pyramids, and mobile heads.

## Related

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/yolo|YOLO]]
- [[papers/architectures/feature-pyramid-networks|Feature Pyramid Networks]]
- [[papers/architectures/retinanet|RetinaNet]]
- [[papers/architectures/faster-r-cnn|Faster R-CNN]]
