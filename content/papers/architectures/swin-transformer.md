---
title: Swin Transformer
aliases:
  - papers/swin-transformer
tags:
  - papers
  - architectures
  - vision
  - transformer
---

# Swin Transformer

> The paper introduced a hierarchical vision Transformer that uses shifted local windows for efficient multi-scale visual representation.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows |
| Authors | Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo |
| Year | 2021 |
| Venue | ICCV 2021 |
| arXiv | [2103.14030](https://arxiv.org/abs/2103.14030) |
| CVF | [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) |
| Status | verified |

## Question

Vision Transformer treats an image as a sequence of fixed-size patches, but dense global attention is expensive for high-resolution images and does not naturally build a multi-scale feature pyramid. The question was how to make Transformer-based vision backbones more compatible with dense prediction tasks.

## Main Claim

A hierarchical Transformer with shifted window attention can provide efficient local attention, cross-window interaction, and multi-scale visual features.

Narrowed claim:

$$
\text{global image attention}
\rightarrow
\text{local window attention}
+
\text{shifted windows}
+
\text{hierarchical patch merging}
$$

This changes the vision Transformer from a flat patch sequence classifier into a general-purpose vision backbone.

## Method

Swin Transformer uses:

| Component | Role |
| --- | --- |
| window self-attention | compute attention locally within windows |
| shifted windows | connect neighboring windows across layers |
| patch merging | reduce resolution and increase channels hierarchically |
| relative position bias | encode local spatial offsets |

Window attention reduces the cost relative to dense attention over all image patches:

$$
O((HW)^2)
\rightarrow
O(HW M^2)
$$

where $M$ is the window size.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Swin works as a general vision backbone | image classification, object detection, and segmentation experiments | vision-specific architecture and training recipe |
| Shifted windows enable cross-window communication | ablations and dense prediction results | not the only possible local-attention design |
| Hierarchical features help dense prediction | COCO and ADE20K results | comparisons depend on backbone and training setup |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | general vision backbone evaluation |
| Input/output unit | image to class, boxes, masks, or segmentation |
| Architecture family | hierarchical vision Transformer |
| Main comparison | CNN backbones and flat vision Transformers |
| Not directly tested | non-image sequence modeling, graph learning, molecular structure modeling |

## Limitations

- Swin reintroduces vision-specific locality and hierarchy, so it is not simply a pure Transformer over all patches.
- Window size, shift pattern, resolution schedule, and training recipe are part of the claim.
- Dense prediction success depends on integration with detection and segmentation heads.
- Later architectures changed local attention, pooling, convolution hybrids, and scaling recipes.

## Why It Matters

Swin Transformer is the anchor paper for hierarchical local-attention vision Transformers and clarifies why vision backbones need more than flat tokenization.

## Connections

- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[papers/architectures/vision-transformer|An Image is Worth 16x16 Words]]
- [[papers/architectures/index|Architecture papers]]
