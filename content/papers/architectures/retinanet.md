---
title: RetinaNet
aliases:
  - papers/retinanet
  - papers/focal-loss-for-dense-object-detection
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - cnn
  - dense-prediction
  - focal-loss
---

# RetinaNet

> The paper shows that a dense one-stage detector can match two-stage detector accuracy when foreground-background imbalance is handled by focal loss.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Focal Loss for Dense Object Detection |
| Authors | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár |
| Year | 2017 |
| Venue | ICCV 2017 |
| arXiv | [1708.02002](https://arxiv.org/abs/1708.02002) |
| Status | seed note started |

## One-Line Takeaway

RetinaNet combines a feature pyramid, dense anchor heads, and focal loss so one-stage detection is not overwhelmed by easy background examples.

## Question

Two-stage detectors reduce class imbalance by evaluating a sparse set of proposals:

$$
\text{image}
\rightarrow
\text{proposals}
\rightarrow
\text{proposal classifier}.
$$

Dense one-stage detectors evaluate many locations and anchors:

$$
\text{image}
\rightarrow
\{\text{all dense candidate boxes}\}.
$$

Most candidates are background. RetinaNet asks:

> Is class imbalance the main reason one-stage detectors trail two-stage detectors, and can the loss fix it?

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| backbone CNN | image | feature hierarchy | visual representation |
| FPN | backbone features | pyramid features $P_l$ | multi-scale dense features |
| classification subnet | each pyramid level | class logits per anchor | dense object classification |
| box subnet | each pyramid level | box offsets per anchor | dense localization |
| focal loss | class logits and labels | reweighted classification loss | focus training on hard examples |

The architecture can be written as:

$$
I
\rightarrow
\{P_l\}
\rightarrow
\{(\hat{p}_{l,i,a},\hat{t}_{l,i,a})\}.
$$

where $l$ is pyramid level, $i$ is spatial location, and $a$ is anchor index.

## FPN Backbone

RetinaNet uses [[papers/architectures/feature-pyramid-networks|Feature Pyramid Networks]]:

$$
\{C_l\}
\rightarrow
\{P_l\}.
$$

Each level receives two subnetworks:

$$
P_l
\rightarrow
\operatorname{ClsSubnet}(P_l),
$$

$$
P_l
\rightarrow
\operatorname{BoxSubnet}(P_l).
$$

The heads are shared across pyramid levels, which makes the detector a repeated dense prediction module over scale.

## Focal Loss

For binary classification, cross entropy is:

$$
\operatorname{CE}(p_t)
=
-\log(p_t),
$$

where:

$$
p_t
=
\begin{cases}
p & \text{if } y=1,\\
1-p & \text{if } y=0.
\end{cases}
$$

Focal loss adds a modulating factor:

$$
\operatorname{FL}(p_t)
=
-
\alpha_t
(1-p_t)^\gamma
\log(p_t).
$$

When an example is already easy, $p_t$ is large and:

$$
(1-p_t)^\gamma
\approx
0.
$$

So easy negatives contribute less, and hard examples dominate training.

## Dense Detector Interpretation

RetinaNet is architecturally simple:

$$
\text{backbone}
+
\text{FPN}
+
\text{two dense heads}.
$$

The paper's important point is that architecture and objective are coupled. A dense one-stage architecture creates many easy background examples; focal loss makes that architecture trainable.

## Comparison

| Axis | [SSD](/papers/architectures/ssd) | RetinaNet |
| --- | --- | --- |
| Detector type | one-stage dense detector | one-stage dense detector |
| Scale handling | multiple feature maps | FPN pyramid |
| Classification issue | hard negative mining | focal loss |
| Main architectural block | default boxes over feature maps | FPN plus shared subnet heads |
| Historical role | practical single-shot detection | closes one-stage/two-stage accuracy gap |

## What To Watch

- The main novelty is focal loss, but the detector architecture is still part of the contribution.
- FPN, anchors, and focal loss interact; avoid attributing all gains to only one component.
- $\gamma$ and $\alpha$ are loss hyperparameters that affect optimization.
- Dense detection accuracy depends on assignment rules and anchor design.
- RetinaNet is not proposal-free in the sense of avoiding anchors; it is proposal-free in the two-stage RPN/RoI sense.

## Related

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/architectures/cnn|CNN]]
- [[papers/architectures/feature-pyramid-networks|Feature Pyramid Networks]]
- [[papers/architectures/ssd|SSD]]
- [[papers/architectures/yolo|YOLO]]
- [[papers/architectures/faster-r-cnn|Faster R-CNN]]
- [[papers/architectures/detr|DETR]]
