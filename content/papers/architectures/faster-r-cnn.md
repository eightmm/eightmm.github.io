---
title: Faster R-CNN
aliases:
  - papers/faster-r-cnn
  - papers/faster-rcnn
  - papers/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - cnn
---

# Faster R-CNN

> The paper makes region proposal generation a learnable neural module shared with the detection network.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks |
| Authors | Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun |
| Year | 2015 |
| Venue | NeurIPS 2015 |
| arXiv | [1506.01497](https://arxiv.org/abs/1506.01497) |
| Status | seed note started |

## One-Line Takeaway

Faster R-CNN turns external region proposal search into a Region Proposal Network, so object proposals and object classification share convolutional image features in one trainable detector.

## Question

Earlier R-CNN-style detectors separated two steps:

1. generate candidate object regions;
2. classify and refine each region.

The proposal step was expensive and not fully integrated with the CNN detector. Faster R-CNN asks:

> Can object proposals be produced by a neural network that shares the same image features used for detection?

The answer is the Region Proposal Network:

$$
I
\rightarrow
F
\rightarrow
\{\text{objectness}, \text{box deltas}\}
\rightarrow
\text{detector head}.
$$

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| CNN backbone | image $I$ | feature map $F \in \mathbb{R}^{C \times H \times W}$ | shared visual representation |
| Region Proposal Network | $F$ | proposal boxes and objectness scores | class-agnostic object candidate generator |
| RoI pooling | $F$ and proposal boxes | fixed-size region features | converts variable boxes to fixed tensors |
| detection head | pooled region features | class scores and box refinements | category prediction and localization |

The key architectural move:

$$
\text{external proposal algorithm}
\rightarrow
\text{learned proposal subnetwork}.
$$

## Region Proposal Network

At each spatial location of the feature map, the RPN evaluates a small set of anchor boxes:

$$
A = \{a_{h,w,k}\},
$$

where $h,w$ index the feature location and $k$ indexes anchor scale/aspect ratio.

For each anchor, the RPN predicts:

$$
p_{h,w,k} = P(\text{object} \mid F_{h,w}),
$$

and box offsets:

$$
t_{h,w,k} = (\Delta x, \Delta y, \Delta w, \Delta h).
$$

A proposal is produced by applying the predicted offsets to the anchor:

$$
\hat{b}_{h,w,k}
=
\operatorname{decode}(a_{h,w,k}, t_{h,w,k}).
$$

The RPN is fully convolutional: it slides over the feature map rather than evaluating each candidate region independently.

## Detection Head

The detector receives proposals from the RPN:

$$
\mathcal{R}=\{r_1,\ldots,r_N\}.
$$

Each region is pooled to a fixed-size tensor:

$$
z_i = \operatorname{RoIPool}(F, r_i).
$$

Then the head predicts a class distribution and box refinement:

$$
\hat{p}_i=\operatorname{softmax}(W_c z_i),
$$

$$
\hat{t}_i=W_b z_i.
$$

This makes the model a two-stage detector:

1. propose object-like regions;
2. classify and refine those regions.

## Multi-Task Loss

The RPN uses objectness and box regression losses:

$$
\mathcal{L}_{\mathrm{RPN}}
=
\frac{1}{N_{\mathrm{cls}}}
\sum_i
\mathcal{L}_{\mathrm{obj}}(p_i,p_i^\*)
+
\lambda
\frac{1}{N_{\mathrm{reg}}}
\sum_i
p_i^\*
\mathcal{L}_{\mathrm{box}}(t_i,t_i^\*).
$$

The detector head uses class and box losses:

$$
\mathcal{L}_{\mathrm{det}}
=
\mathcal{L}_{\mathrm{cls}}(\hat{p}, c^\*)
+
\lambda
\mathbf{1}[c^\*>0]
\mathcal{L}_{\mathrm{box}}(\hat{t}, t^\*).
$$

The foreground indicator prevents background regions from contributing to box regression.

## Why It Matters

Faster R-CNN is not just a better detector. It defines a reusable detection architecture pattern:

$$
\text{backbone}
+
\text{proposal module}
+
\text{region head}.
$$

That pattern underlies later instance segmentation and keypoint models, especially [[papers/architectures/mask-r-cnn|Mask R-CNN]].

For modern reading, it is useful because it contrasts sharply with [[papers/architectures/detr|DETR]]:

| Axis | Faster R-CNN | DETR |
| --- | --- | --- |
| Candidate objects | anchors and proposals | learned object queries |
| Assignment | proposal sampling and IoU heuristics | bipartite matching |
| Duplicate handling | non-maximum suppression | set prediction loss |
| Inductive bias | local CNN features plus region proposals | global Transformer interaction |

## What To Watch

- Proposal quality and detection quality are coupled but not identical.
- Anchor design changes the search space and can affect small-object recall.
- Non-maximum suppression is still a post-processing dependency.
- Speed claims depend on backbone, proposal count, image size, and hardware.
- The architecture is strong for object-level localization but not designed for dense masks by itself.

## Related

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/localization|Localization]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/mask-r-cnn|Mask R-CNN]]
- [[papers/architectures/detr|DETR]]
