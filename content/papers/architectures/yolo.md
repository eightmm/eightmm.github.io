---
title: YOLO
aliases:
  - papers/yolo
  - papers/you-only-look-once
  - papers/you-only-look-once-unified-real-time-object-detection
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - cnn
  - real-time-detection
---

# YOLO

> The paper frames object detection as a single regression problem from a full image to boxes and class probabilities.

## Metadata

| Field | Value |
| --- | --- |
| Paper | You Only Look Once: Unified, Real-Time Object Detection |
| Authors | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi |
| Year | 2016 |
| Venue | CVPR 2016 |
| arXiv | [1506.02640](https://arxiv.org/abs/1506.02640) |
| Paper | [CVF Open Access PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) |
| Status | seed note started |

## One-Line Takeaway

YOLO replaces proposal-based detection with a single CNN that predicts bounding boxes and class probabilities directly from the full image in one forward pass.

## Question

Earlier detection pipelines often looked like:

$$
\text{region proposals}
\rightarrow
\text{classification}
\rightarrow
\text{box refinement}
\rightarrow
\text{post-processing}.
$$

YOLO asks:

> Can object detection be trained as one direct prediction problem?

The proposed interface:

$$
I
\rightarrow
\hat{Y},
$$

where:

$$
\hat{Y}
\in
\mathbb{R}^{S\times S\times (B\cdot5+C)}.
$$

Here:

- $S\times S$ is the image grid;
- $B$ is the number of boxes per grid cell;
- $C$ is the number of classes;
- each box predicts $(x,y,w,h,\text{confidence})$.

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| CNN backbone | image | spatial feature grid | full-image visual representation |
| grid predictor | feature grid | box and class predictions per cell | single-stage dense prediction |
| detection loss | predicted grid and targets | scalar objective | train boxes, objectness, and classes together |
| NMS/post-process | candidate boxes | final detections | remove duplicate boxes |

The main architectural shift:

$$
\text{proposal-classification pipeline}
\rightarrow
\text{single full-image predictor}.
$$

## Grid Prediction

YOLO divides the image into an $S\times S$ grid. Each grid cell is responsible for objects whose center falls inside that cell.

For cell $(i,j)$ and box $b$, the model predicts:

$$
\hat{y}_{i,j,b}
=
(\hat{x},\hat{y},\hat{w},\hat{h},\hat{c}).
$$

The confidence combines objectness and localization quality:

$$
\hat{c}
\approx
P(\text{object})
\cdot
\operatorname{IoU}(\hat{b}, b^\*).
$$

Class probabilities are predicted per cell:

$$
\hat{p}_{i,j,k}
=
P(c=k \mid \text{object in cell } i,j).
$$

The final class-specific score is:

$$
s_{i,j,b,k}
=
\hat{c}_{i,j,b}
\cdot
\hat{p}_{i,j,k}.
$$

## Unified Detection Loss

A simplified YOLO-style objective combines localization, confidence, and classification:

$$
\mathcal{L}
=
\lambda_{\mathrm{coord}}\mathcal{L}_{\mathrm{box}}
+
\mathcal{L}_{\mathrm{conf}}
+
\mathcal{L}_{\mathrm{cls}}.
$$

The original formulation weights coordinate error and no-object confidence differently:

$$
\mathcal{L}_{\mathrm{box}}
=
\sum_{i,j,b}
\mathbf{1}_{i,j,b}^{\mathrm{obj}}
\left[
(x-\hat{x})^2
+
(y-\hat{y})^2
+
(\sqrt{w}-\sqrt{\hat{w}})^2
+
(\sqrt{h}-\sqrt{\hat{h}})^2
\right].
$$

The square-root width/height terms reduce the penalty imbalance between small and large boxes.

## Why It Matters

YOLO is the canonical one-stage real-time detector starting point:

$$
\text{one image}
\rightarrow
\text{one network evaluation}
\rightarrow
\text{detections}.
$$

It is architecturally important because it makes speed and simplicity part of the model contract. The design sacrifices some localization precision compared with two-stage detectors, but it makes detection usable in real-time systems.

## Comparison

| Axis | [Faster R-CNN](/papers/architectures/faster-r-cnn) | YOLO |
| --- | --- | --- |
| Detection style | two-stage | one-stage |
| Candidate generation | RPN proposals | grid predictions |
| Speed target | accurate proposal detector | real-time detection |
| Context | region-centric | full-image |
| Main weakness | heavier pipeline | localization errors, small objects |

## Reading Boundary

YOLO has many later versions. This note is about the original YOLOv1 architecture paper. Later YOLO variants add anchors, stronger backbones, feature pyramids, better assignment rules, and many training changes. Those should not be retroactively attributed to the original paper.

## What To Watch

- Grid-cell responsibility can struggle when multiple small objects fall in the same cell.
- Direct regression is fast but can trade away localization precision.
- The final detector still uses post-processing such as non-maximum suppression.
- Later YOLO versions differ substantially from YOLOv1.
- Benchmark comparisons should separate speed, input resolution, backbone, and dataset.

## Related

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/feature-pyramid-networks|Feature Pyramid Networks]]
- [[papers/architectures/faster-r-cnn|Faster R-CNN]]
- [[papers/architectures/detr|DETR]]
