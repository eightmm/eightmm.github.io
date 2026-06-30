---
title: EfficientNet
aliases:
  - papers/efficientnet
  - papers/rethinking-model-scaling-for-convolutional-neural-networks
tags:
  - papers
  - architectures
  - cnn
  - vision
  - scaling
---

# EfficientNet

> The paper introduced compound scaling for CNN depth, width, and input resolution.

## Metadata

| Field | Value |
| --- | --- |
| Paper | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks |
| Authors | Mingxing Tan, Quoc V. Le |
| Year | 2019 |
| Venue | ICML 2019 |
| arXiv | [1905.11946](https://arxiv.org/abs/1905.11946) |
| PMLR | [v97/tan19a](https://proceedings.mlr.press/v97/tan19a.html) |
| Status | verified |

## Question

CNNs can be scaled by making them deeper, wider, or by using larger input resolution. The question was whether those dimensions should be scaled together under a principled compute budget.

## Main Claim

EfficientNet scales depth, width, and resolution with one compound coefficient:

$$
\begin{aligned}
d &= \alpha^\phi \\
w &= \beta^\phi \\
r &= \gamma^\phi
\end{aligned}
$$

subject to an approximate compute constraint:

$$
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

where $\phi$ controls the overall resource scale.

## Method

| Component | Role |
| --- | --- |
| baseline network search | finds EfficientNet-B0 |
| compound scaling | balances depth, width, and resolution |
| MBConv-style blocks | efficient convolutional building blocks |
| model family | B0 through larger scaled variants |
| transfer evaluation | checks whether scaling transfers beyond ImageNet |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Balanced scaling improves accuracy/efficiency | ImageNet accuracy, parameters, and FLOPs comparisons | baseline found by NAS, not hand-designed |
| Scaling all dimensions beats one-axis scaling | ablation on depth/width/resolution scaling | result depends on chosen baseline and search space |
| EfficientNets transfer well | CIFAR, Flowers, and other transfer datasets | transfer does not cover every vision task |

## Limitations

- EfficientNet is partly a scaling rule and partly a NAS-discovered model family.
- FLOPs and parameter count do not fully predict latency on real hardware.
- Training recipe and input resolution are central to the result.
- Later architectures changed augmentation, normalization, convolution blocks, and deployment constraints.

## Why It Matters

EfficientNet is the canonical CNN paper for architecture scaling under compute constraints.

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/inception|Inception]]
- [[papers/architectures/index|Architecture papers]]
