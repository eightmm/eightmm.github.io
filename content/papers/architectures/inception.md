---
title: Going Deeper with Convolutions
aliases:
  - papers/inception
  - papers/going-deeper-with-convolutions
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# Going Deeper with Convolutions

> The paper introduced the Inception module as a multi-branch CNN block for increasing depth and width under a compute budget.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Going Deeper with Convolutions |
| Authors | Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich |
| Year | 2015 |
| Venue | CVPR 2015 |
| arXiv | [1409.4842](https://arxiv.org/abs/1409.4842) |
| CVF | [CVPR 2015 paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) |
| Status | verified |

## Question

CNNs were getting deeper and wider, but naive scaling increased parameters and compute. The question was whether a CNN block could use multiple receptive-field sizes while keeping computation practical.

## Main Claim

An Inception module can approximate multi-scale feature extraction by combining parallel convolution and pooling branches, then concatenating channels.

Abstract module form:

$$
y
=
\operatorname{concat}
\left[
f_{1 \times 1}(x),
f_{3 \times 3}(x),
f_{5 \times 5}(x),
p(x)
\right]
$$

with $1 \times 1$ convolutions used to reduce channel dimension before expensive branches.

## Method

| Component | Role |
| --- | --- |
| parallel branches | represent multiple receptive-field scales |
| $1 \times 1$ convolution | channel projection and compute control |
| concatenation | merges branch outputs |
| auxiliary classifiers | improve training signal in deep networks |
| global average pooling | reduces classifier parameter count |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Inception improves ImageNet classification/detection | ILSVRC 2014 results | architecture and training system are intertwined |
| Multi-branch modules use compute efficiently | parameter/operation-aware design | later Inception variants changed many details |
| $1 \times 1$ projections make wider modules practical | module design and empirical performance | not isolated from all training choices |

## Limitations

- The module has many design choices and branch widths, making it less simple than VGG.
- Later residual and normalization designs changed the default CNN recipe.
- The architecture is specialized for image grids.
- The paper is a system milestone, so evidence mixes architecture, training, and ensemble choices.

## Why It Matters

Inception is the canonical paper for multi-branch CNN modules, compute-aware width/depth scaling, and $1 \times 1$ convolution as architectural projection.

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[papers/architectures/vgg|VGG]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/index|Architecture papers]]
