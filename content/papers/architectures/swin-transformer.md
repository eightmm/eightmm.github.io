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

The deeper question is whether a vision Transformer should remain a flat global-token model. CNN backbones naturally create hierarchical feature maps. Object detection and segmentation often expect multi-resolution features. Swin asks whether a Transformer can recover that hierarchy while keeping attention efficient.

This reframes the ViT design:

$$
\text{flat global patch Transformer}
\rightarrow
\text{hierarchical local-window Transformer}
$$

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

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor |
| Initial representation | non-overlapping patch tokens |
| Main token mixing | local window self-attention |
| Cross-window communication | shifted window partition across alternating blocks |
| Hierarchy | patch merging reduces resolution and increases channels |
| Position signal | relative position bias inside windows |
| Natural tasks | classification, detection, segmentation |

If the feature map has spatial size $H \times W$ and channel dimension $C$, Swin keeps a 2D token map rather than collapsing the image into a purely flat sequence for the entire backbone:

$$
X
\in
\mathbb{R}^{H \times W \times C}
$$

The model repeatedly applies local attention and spatial downsampling:

$$
(H,W,C)
\rightarrow
\left(\frac{H}{2}, \frac{W}{2}, 2C\right)
\rightarrow
\left(\frac{H}{4}, \frac{W}{4}, 4C\right)
\rightarrow
\cdots
$$

This is why Swin behaves more like a general-purpose vision backbone than the original flat ViT classifier.

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

## Window Self-Attention

Instead of applying attention over all $HW$ image tokens, Swin partitions the feature map into local windows. If each window has $M \times M$ tokens, attention is computed within each window:

$$
\operatorname{W\text{-}MSA}(X)
=
\{\operatorname{MSA}(X_w)\}_{w=1}^{N_w}
$$

where $X_w$ is one window and $N_w$ is the number of windows.

Dense global attention over all image tokens has cost:

$$
O((HW)^2 C)
$$

Window attention has approximate cost:

$$
O(HW M^2 C)
$$

when $M$ is fixed. This makes the attention cost linear in image area for fixed window size.

## Shifted Windows

Local windows alone isolate neighboring windows. Swin alternates regular windows and shifted windows so tokens can interact across previous window boundaries.

Conceptually:

$$
\text{block } l:
\text{ window partition}
$$

$$
\text{block } l+1:
\text{ shifted window partition}
$$

The shift means a token's local attention neighborhood changes across blocks. Cross-window communication emerges without paying full global attention cost.

| Design | Benefit | Cost |
| --- | --- | --- |
| fixed local windows | efficient local attention | windows are isolated |
| shifted windows | cross-window interaction | mask and partition complexity |
| global attention | direct all-pair interaction | quadratic cost |

## Patch Merging and Hierarchy

Swin builds stages. Between stages, patch merging reduces spatial resolution and increases channel dimension.

Patch merging concatenates neighboring tokens:

$$
[x_{2i,2j}, x_{2i+1,2j}, x_{2i,2j+1}, x_{2i+1,2j+1}]
$$

then projects the concatenated vector:

$$
x'_{i,j}
=
W
[x_{2i,2j}; x_{2i+1,2j}; x_{2i,2j+1}; x_{2i+1,2j+1}]
$$

This creates a multi-scale feature hierarchy similar in spirit to CNN backbones.

| Stage Behavior | Vision Meaning |
| --- | --- |
| high resolution, low channels | local edges/textures/small objects |
| lower resolution, higher channels | semantic regions and object-level features |
| multi-scale outputs | detection and segmentation compatibility |

## Relative Position Bias

Inside each local window, Swin adds a relative position bias to attention logits:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + B
\right)V
$$

where $B$ depends on the relative offset between tokens inside the window.

This matters because local spatial relations are central to vision. Swin does not rely only on absolute position embeddings from the original ViT recipe.

## Relation to ViT

| Axis | ViT | Swin |
| --- | --- | --- |
| token layout | flat patch sequence | hierarchical 2D token maps |
| attention | global attention | local shifted-window attention |
| cost with resolution | quadratic in patch count | linear in image area for fixed window size |
| dense prediction fit | needs adaptation | designed as general backbone |
| image bias | weaker | reintroduces locality and hierarchy |

Swin can be read as a correction to the original ViT design for vision infrastructure. It keeps Transformer blocks but restores some of the structure that made CNN backbones useful.

## Architecture Tradeoff

Swin gives up immediate global all-token attention in exchange for efficient local attention and staged hierarchy.

$$
\text{global flexibility}
\downarrow
\quad
\text{local efficiency and hierarchy}
\uparrow
$$

This tradeoff is acceptable when:

- high-resolution images make dense attention expensive;
- local visual neighborhoods are meaningful;
- downstream tasks need multi-scale feature maps;
- detection/segmentation heads expect backbone pyramids.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Swin works as a general vision backbone | image classification, object detection, and segmentation experiments | vision-specific architecture and training recipe |
| Shifted windows enable cross-window communication | ablations and dense prediction results | not the only possible local-attention design |
| Hierarchical features help dense prediction | COCO and ADE20K results | comparisons depend on backbone and training setup |

## Benchmark Reading

Swin should be read as a backbone paper, not only an ImageNet classifier.

| Benchmark Type | Why it matters |
| --- | --- |
| ImageNet classification | checks classification backbone quality |
| COCO detection | tests object-level dense prediction |
| ADE20K segmentation | tests dense semantic prediction |
| throughput/complexity | checks whether local windows actually help |

The strongest claim is that Swin works across classification, detection, and segmentation with one hierarchical Transformer backbone.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | general vision backbone evaluation |
| Input/output unit | image to class, boxes, masks, or segmentation |
| Architecture family | hierarchical vision Transformer |
| Main comparison | CNN backbones and flat vision Transformers |
| Not directly tested | non-image sequence modeling, graph learning, molecular structure modeling |

## Ablation Reading

| Axis | What it tests | Reading |
| --- | --- | --- |
| shifted vs non-shifted windows | cross-window communication | central to the paper's design |
| window size | locality/compute tradeoff | too small limits context, too large increases cost |
| hierarchy/patch merging | dense prediction compatibility | makes Swin backbone-like |
| relative position bias | local spatial relation modeling | important for vision attention |
| task transfer | classification vs detection vs segmentation | tests general backbone claim |

The key ablation question is whether shifted windows recover enough cross-window information without returning to global attention.

## Implementation Notes

- Window partitioning and reverse partitioning must preserve spatial layout exactly.
- Shifted windows require attention masks so wrapped tokens do not attend incorrectly.
- Relative position bias tables are indexed by window-relative offsets.
- Feature pyramid integration matters for detection and segmentation results.
- Window size interacts with input resolution, memory, and throughput.
- Swin is not a pure ViT; it deliberately reintroduces vision-specific structure.

## Limitations

- Swin reintroduces vision-specific locality and hierarchy, so it is not simply a pure Transformer over all patches.
- Window size, shift pattern, resolution schedule, and training recipe are part of the claim.
- Dense prediction success depends on integration with detection and segmentation heads.
- Later architectures changed local attention, pooling, convolution hybrids, and scaling recipes.
- Local windows can miss long-range interactions unless depth and shifts propagate information sufficiently.
- Implementation complexity is higher than flat ViT.
- Claims outside image grids need separate evidence.

## Why It Matters

Swin Transformer is the anchor paper for hierarchical local-attention vision Transformers and clarifies why vision backbones need more than flat tokenization.

The reusable architecture pattern is:

$$
\text{local attention}
+
\text{shifted communication}
+
\text{hierarchical merging}
\rightarrow
\text{Transformer backbone for dense vision}
$$

For this wiki, Swin marks the point where Transformer vision models stop being only patch-sequence classifiers and become practical backbones.

## Connections

- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[papers/architectures/vision-transformer|An Image is Worth 16x16 Words]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/index|Architecture papers]]
