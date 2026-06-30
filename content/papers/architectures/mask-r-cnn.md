---
title: Mask R-CNN
aliases:
  - papers/mask-r-cnn
  - papers/mask-rcnn
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - segmentation
  - cnn
---

# Mask R-CNN

> The paper extends Faster R-CNN with a parallel mask prediction branch and a better feature alignment operator for instance segmentation.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Mask R-CNN |
| Authors | Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick |
| Year | 2017 |
| Venue | ICCV 2017 |
| arXiv | [1703.06870](https://arxiv.org/abs/1703.06870) |
| Status | seed note started |

## One-Line Takeaway

Mask R-CNN keeps the two-stage detection structure of [[papers/architectures/faster-r-cnn|Faster R-CNN]] and adds an instance-specific dense mask head, making detection, box regression, and mask prediction share one region-based architecture.

## Question

Object detection predicts boxes:

$$
\hat{Y}_{\mathrm{det}}
=
\{(\hat{c}_i,\hat{b}_i,\hat{s}_i)\}_{i=1}^{N}.
$$

Instance segmentation predicts boxes plus a separate mask for each object:

$$
\hat{Y}_{\mathrm{inst}}
=
\{(\hat{c}_i,\hat{b}_i,\hat{m}_i,\hat{s}_i)\}_{i=1}^{N},
$$

where:

- $\hat{c}_i$ is the class;
- $\hat{b}_i$ is the bounding box;
- $\hat{m}_i \in [0,1]^{H_m \times W_m}$ is the instance mask;
- $\hat{s}_i$ is the confidence score.

The paper asks:

> Can a strong two-stage detector be extended to pixel-level instance masks without making the architecture complicated?

## Architecture Contract

| Component | Role |
| --- | --- |
| CNN backbone | extracts image features |
| Region Proposal Network | proposes object-like regions |
| RoIAlign | extracts spatially aligned region features |
| classification head | predicts object category |
| box head | refines bounding box coordinates |
| mask head | predicts a class-specific binary mask for each RoI |

The architectural change from Faster R-CNN:

$$
\text{region detection head}
\rightarrow
\text{region detection head}
+
\text{parallel mask head}.
$$

## RoIAlign

Faster R-CNN uses RoI pooling to convert variable-sized regions into fixed-size tensors. Quantization in this step can misalign features and pixels, which hurts mask quality.

Mask R-CNN replaces it with RoIAlign:

$$
z_i
=
\operatorname{RoIAlign}(F, r_i),
$$

where $F$ is the feature map and $r_i$ is a proposed region.

Conceptually:

1. avoid rounding region coordinates to coarse feature bins;
2. sample feature values at fractional coordinates;
3. use bilinear interpolation;
4. preserve spatial alignment for the mask head.

For a fractional coordinate $(x,y)$, bilinear interpolation can be written as:

$$
F(x,y)
=
\sum_{u,v}
w_{uv}(x,y)F(u,v),
$$

where neighboring grid points $(u,v)$ contribute according to interpolation weights $w_{uv}$.

This is a small architectural operator, but it matters because masks are evaluated at pixel-level boundaries.

## Mask Head

For each region feature $z_i$, Mask R-CNN predicts:

$$
\hat{m}_i
=
f_{\mathrm{mask}}(z_i)
\in
[0,1]^{K \times H_m \times W_m},
$$

where $K$ is the number of classes.

The selected mask is usually the channel corresponding to the predicted class:

$$
\hat{m}_{i,\hat{c}_i}
\in
[0,1]^{H_m \times W_m}.
$$

The mask branch is fully convolutional, so it preserves spatial structure inside the RoI:

$$
z_i
\rightarrow
\operatorname{Conv}
\rightarrow
\operatorname{Conv}
\rightarrow
\hat{m}_i.
$$

## Multi-Task Objective

Mask R-CNN uses the Faster R-CNN detection losses plus a mask loss:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{cls}}
+
\mathcal{L}_{\mathrm{box}}
+
\mathcal{L}_{\mathrm{mask}}.
$$

For the target class $k^\*$, the mask loss is binary cross entropy over the mask grid:

$$
\mathcal{L}_{\mathrm{mask}}
=
-
\frac{1}{H_mW_m}
\sum_{h,w}
\left[
m^\*_{h,w}\log \hat{m}_{k^\*,h,w}
+
(1-m^\*_{h,w})
\log(1-\hat{m}_{k^\*,h,w})
\right].
$$

This keeps mask prediction class-specific while avoiding competition between classes at every pixel.

## Why It Matters

Mask R-CNN is a canonical example of extending an architecture by adding a task head while preserving the core representation pipeline:

$$
\text{shared image features}
\rightarrow
\text{region features}
\rightarrow
\{\text{class}, \text{box}, \text{mask}\}.
$$

The important lesson is not only that masks improve segmentation. It is that a clean architecture can support several structured outputs if the intermediate representation has the right granularity.

## Relation To Later Models

| Model family | Object representation | Output style |
| --- | --- | --- |
| Faster R-CNN | proposed RoIs | class and box |
| Mask R-CNN | proposed RoIs | class, box, and mask |
| DETR | learned object queries | class and box set |
| modern segmenters | prompts, masks, or query slots | dense masks or mask sets |

Mask R-CNN is still useful to read before modern promptable segmentation systems because it makes the difference between detection boxes, aligned RoI features, and per-instance masks explicit.

## What To Watch

- Mask quality depends heavily on feature alignment.
- The model inherits two-stage detector dependencies: proposals, anchor choices, and NMS.
- The mask head predicts masks per region, not a single global semantic map.
- Box AP and mask AP can move differently; one does not fully explain the other.
- The method is architecture-general: stronger backbones and feature pyramids can be plugged in.

## Related

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/localization|Localization]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/feature-pyramid-networks|Feature Pyramid Networks]]
- [[papers/architectures/faster-r-cnn|Faster R-CNN]]
- [[papers/architectures/detr|DETR]]
