---
title: DETR
aliases:
  - papers/detr
  - papers/end-to-end-object-detection-with-transformers
tags:
  - papers
  - architectures
  - computer-vision
  - object-detection
  - transformer
---

# DETR

> The paper reframes object detection as direct set prediction with a Transformer encoder-decoder and bipartite matching loss.

## Metadata

| Field | Value |
| --- | --- |
| Paper | End-to-End Object Detection with Transformers |
| Authors | Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko |
| Year | 2020 |
| Venue | ECCV 2020 |
| arXiv | [2005.12872](https://arxiv.org/abs/2005.12872) |
| Paper | [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf) |
| Official implementation | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| Status | full note started |

## One-Line Takeaway

DETR removes anchors, proposal stages, and non-maximum suppression by making object detection a set prediction problem solved with learned object queries and Hungarian matching.

## Question

Traditional object detection pipelines often include hand-designed components:

- anchors;
- region proposals;
- non-maximum suppression;
- heuristic matching rules;
- task-specific post-processing.

The output of object detection is a set:

$$
Y = \{(c_i,b_i)\}_{i=1}^{M},
$$

where $c_i$ is a class and $b_i$ is a bounding box.

Sets have no canonical order:

$$
\{a,b,c\} = \{b,c,a\}.
$$

The paper asks:

> Can object detection be trained end-to-end as direct set prediction?

DETR's answer:

$$
\text{image}
\rightarrow
\text{fixed number of object query predictions}
\rightarrow
\text{set matching loss}.
$$

## Main Claim

Object detection can be simplified by combining:

1. a CNN image backbone;
2. a Transformer encoder-decoder;
3. learned object queries;
4. bipartite matching between predictions and ground truth;
5. a set loss with a no-object class.

The model predicts a fixed-size set:

$$
\hat{Y}
=
\{(\hat{p}_i,\hat{b}_i)\}_{i=1}^{N},
$$

where:

- $N$ is the number of object queries;
- $\hat{p}_i$ is a class distribution including no-object;
- $\hat{b}_i$ is a bounding box.

If an image has $M$ ground-truth objects and $M<N$, the remaining predictions should be no-object.

## Architecture Contract

| Component | Role |
| --- | --- |
| CNN backbone | extracts image feature map |
| positional encoding | preserves spatial information |
| Transformer encoder | globally contextualizes image features |
| object queries | learned slots for possible objects |
| Transformer decoder | maps object queries to object predictions |
| Hungarian matching | assigns predictions to ground-truth objects |
| set loss | trains unique predictions without NMS |

The architectural shift:

$$
\text{dense anchors/proposals}
\rightarrow
\text{learned object query slots}.
$$

## Image Backbone

The image is first encoded by a convolutional backbone:

$$
I
\rightarrow
F \in \mathbb{R}^{C\times H\times W}.
$$

The spatial feature map is flattened into a sequence:

$$
X \in \mathbb{R}^{HW \times C}.
$$

Because flattening destroys explicit 2D coordinates, positional encodings are added:

$$
Z_0 = X + P.
$$

Then the Transformer encoder processes image tokens:

$$
Z = \operatorname{Encoder}(Z_0).
$$

The CNN provides local visual features; the Transformer provides global interaction.

## Transformer Encoder

The encoder applies self-attention over spatial feature tokens:

$$
\operatorname{SelfAttn}(Z)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
$$

where:

$$
Q=ZW^Q,\qquad K=ZW^K,\qquad V=ZW^V.
$$

This lets each spatial location attend to the whole image. The model can reason about object relations and global context, not only local receptive fields.

## Object Queries

DETR uses a fixed set of learned object queries:

$$
Q_{\text{obj}}
=
\{q_1,\dots,q_N\},
\qquad
q_i\in\mathbb{R}^{d}.
$$

Each query is a slot that can become one object prediction.

The decoder maps:

$$
(Q_{\text{obj}}, Z)
\rightarrow
O \in \mathbb{R}^{N\times d}.
$$

Each output slot $o_i$ predicts:

$$
\hat{p}_i = \operatorname{softmax}(W_c o_i),
$$

$$
\hat{b}_i = \operatorname{MLP}(o_i).
$$

The object queries are not input image patches. They are learned prediction slots.

## Transformer Decoder

The decoder has two attention mechanisms:

1. self-attention among object queries;
2. cross-attention from queries to image features.

Query self-attention:

$$
Q' = \operatorname{SelfAttn}(Q_{\text{obj}}).
$$

Cross-attention:

$$
\operatorname{CrossAttn}(Q', Z)
=
\operatorname{softmax}
\left(
\frac{(Q'W^Q)(ZW^K)^\top}{\sqrt{d}}
\right)
ZW^V.
$$

This lets object slots coordinate with each other and attend to relevant image regions.

The self-attention between object queries helps avoid duplicate predictions:

$$
\text{query slots can communicate before final boxes are emitted}.
$$

## Set Prediction

The ground truth is a set:

$$
Y = \{y_i\}_{i=1}^{M}.
$$

The prediction is a fixed-size set:

$$
\hat{Y} = \{\hat{y}_j\}_{j=1}^{N}.
$$

The model needs a one-to-one assignment between ground truth objects and prediction slots. DETR uses bipartite matching:

$$
\hat{\sigma}
=
\arg\min_{\sigma\in\mathfrak{S}_N}
\sum_{i=1}^{M}
\mathcal{C}(y_i,\hat{y}_{\sigma(i)}).
$$

Here $\mathcal{C}$ is a matching cost based on class and box quality.

This is the core reason DETR does not need NMS. Duplicate predictions are penalized by one-to-one matching.

## Matching Cost

A simplified matching cost is:

$$
\mathcal{C}(y_i,\hat{y}_j)
=
-\log \hat{p}_j(c_i)
+
\lambda_{\ell_1}
\lVert b_i-\hat{b}_j\rVert_1
+
\lambda_{\text{giou}}
\mathcal{L}_{\text{GIoU}}(b_i,\hat{b}_j).
$$

where:

- $c_i$ is ground-truth class;
- $b_i$ is ground-truth box;
- $\hat{p}_j$ is predicted class distribution;
- $\hat{b}_j$ is predicted box;
- GIoU measures box overlap quality.

After matching, the training loss is computed on matched pairs plus no-object targets for unmatched slots.

## Training Loss

For matched ground-truth/prediction pairs:

$$
L_{\text{matched}}
=
\sum_i
\left[
L_{\text{class}}(c_i,\hat{p}_{\hat{\sigma}(i)})
+
\lambda_{\ell_1}
\lVert b_i-\hat{b}_{\hat{\sigma}(i)}\rVert_1
+
\lambda_{\text{giou}}
L_{\text{GIoU}}(b_i,\hat{b}_{\hat{\sigma}(i)})
\right].
$$

For unmatched predictions:

$$
L_{\varnothing}
=
\sum_{j\notin \operatorname{matched}}
L_{\text{class}}(\varnothing,\hat{p}_j).
$$

Total loss:

$$
L = L_{\text{matched}} + L_{\varnothing}.
$$

This creates a direct set-level supervision signal.

## Why NMS Is Not Needed

Traditional detectors may produce many overlapping boxes and then remove duplicates with non-maximum suppression.

DETR avoids this by training prediction slots with one-to-one matching:

$$
\text{one ground-truth object}
\leftrightarrow
\text{one prediction slot}.
$$

If two slots predict the same object, only one can be matched to it. The other is pushed toward no-object or penalized through the loss.

The decoder query interaction also lets slots coordinate.

## Comparison to Faster R-CNN Style Detectors

| Property | [Faster R-CNN](/papers/architectures/faster-r-cnn) style detector | DETR |
| --- | --- | --- |
| Candidate generation | anchors/proposals | learned object queries |
| Duplicate removal | NMS | bipartite matching loss |
| Output form | many dense candidates | fixed-size prediction set |
| Architecture | CNN plus detection heads | CNN plus Transformer encoder-decoder |
| Inductive bias | strong spatial detection priors | weaker, more generic set prediction |
| Training | mature and efficient | slower convergence in original DETR |

DETR simplifies the conceptual pipeline, but it pays with training cost and small-object difficulty in the original version.

## Comparison to Set Transformer

[[papers/architectures/set-transformer|Set Transformer]] models unordered input/output structure with attention and learned pooling/query slots. DETR also uses learned slots, but for object detection:

$$
\text{object queries}
\approx
\text{prediction slots}.
$$

| Property | Set Transformer | DETR |
| --- | --- | --- |
| Main domain | generic set learning | object detection |
| Learned queries | PMA seeds/output slots | object queries |
| Loss | task-dependent | Hungarian set loss |
| Output | set-level or slot output | boxes and classes |
| Key issue | permutation symmetry | duplicate-free detection |

DETR is an important example of set prediction becoming a concrete vision architecture.

## Comparison to Vision Transformer

[[papers/architectures/vision-transformer|ViT]] treats image patches as tokens for classification. DETR uses image feature tokens plus object queries for detection.

| Property | ViT | DETR |
| --- | --- | --- |
| Main input tokens | image patches | CNN feature map tokens |
| Output | class token or pooled class | set of object boxes/classes |
| Decoder | not central in original ViT | central |
| Object slots | no | learned object queries |
| Loss | classification | set matching and box loss |

DETR is not simply ViT for detection. Its key novelty is the set decoder and matching objective.

## Evidence Reading

The paper demonstrates that object detection can be handled end-to-end with a Transformer set prediction architecture. The main architectural evidence is conceptual simplification with competitive COCO results.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Detection can be direct set prediction | COCO object detection results | anchors/NMS are not mandatory | original DETR trains slowly |
| Object queries learn prediction slots | decoder attention analysis and performance | learned slots can represent objects | query semantics are not always stable |
| Bipartite matching removes duplicates | set loss formulation | one-to-one matching replaces NMS | matching cost design matters |
| Unified extension to panoptic segmentation | segmentation variant | architecture is flexible | not always best specialized segmentation model |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | object detection and panoptic segmentation |
| Input unit | image |
| Output unit | set of boxes, classes, optional masks |
| Architecture family | CNN plus Transformer encoder-decoder |
| Core mechanism | object queries and Hungarian matching |
| Main comparison | Faster R-CNN style detectors |
| Key issue | slow convergence and small-object performance in original version |
| Not the claim | Transformer attention alone solves all detection efficiency issues |

## Implementation Notes

### Number of Queries

The number of object queries $N$ is the maximum number of predictions:

$$
\hat{Y} = \{\hat{y}_1,\dots,\hat{y}_N\}.
$$

If $N$ is too small, the model cannot predict all objects. If too large, many slots must learn no-object.

### No-Object Class

Each prediction has a no-object class:

$$
\varnothing.
$$

This is essential because most query slots do not correspond to objects in most images.

### Matching Before Loss

The Hungarian assignment is computed before the final supervised loss:

$$
\hat{\sigma}
=
\operatorname{HungarianMatch}(Y,\hat{Y}).
$$

The matching cost and the training loss are related but not necessarily identical.

### Small Objects

Original DETR struggles more with small objects partly because feature resolution and global attention do not provide the same multi-scale priors as FPN-style detectors.

### Convergence

Original DETR needs long training compared to mature detectors. Later variants such as Deformable DETR address this with sparse deformable attention and multi-scale features.

## Molecular and Scientific Modeling Reading

DETR is a vision paper, but the set prediction pattern is relevant outside vision:

$$
\text{input context}
\rightarrow
\text{set of structured outputs}.
$$

Possible analogues:

- predicting a set of binding-site residues;
- predicting multiple ligand poses;
- detecting objects in microscopy images;
- extracting entities from scientific figures;
- predicting candidate interaction pairs;
- set-valued structure annotations.

The useful idea is not bounding boxes themselves. It is:

$$
\text{learned query slots}
+
\text{one-to-one matching loss}.
$$

For molecular structure tasks, the equivalent output might be coordinates, atom groups, interaction sites, or candidate complexes rather than boxes.

## Failure Modes

### Query Collapse or Duplication

Object queries can learn redundant behavior if matching and no-object weighting are poorly tuned.

### Slow Training

The original design is conceptually simple but not training-efficient.

### Small-Object Weakness

Without strong multi-scale design, small objects can be missed.

### Matching Cost Sensitivity

Changing the cost weights:

$$
\lambda_{\ell_1},\lambda_{\text{giou}}
$$

can change assignment and training behavior.

### Weak Local Prior

Dense detectors encode many spatial priors. DETR removes many of them, which improves simplicity but can reduce sample efficiency.

## Common Misreadings

### "DETR is just a Transformer detector."

The essential idea is not only using a Transformer. It is object detection as set prediction with object queries and bipartite matching.

### "No NMS means no post-processing at all."

There is still decoding and thresholding, but duplicate removal via NMS is not central to the method.

### "Object queries correspond to fixed semantic classes."

Not necessarily. They are learned slots, not class prototypes.

### "DETR made all classical detection design obsolete."

No. Many later detectors reintroduce multi-scale features, sparse attention, denoising, anchors-like priors, or hybrid designs for efficiency.

## Later-Paper Checklist

When reading later detection Transformer papers, ask:

- Are object queries learned, dynamic, denoising, or anchor-like?
- Is matching Hungarian, one-to-many, or hybrid?
- Is NMS truly removed?
- How many queries are used?
- How is no-object handled?
- Are multi-scale features used?
- Is attention dense, deformable, sparse, or local?
- How fast does it converge?
- How does it perform on small objects?
- Are comparisons made at equal schedule and augmentation?

## Why It Matters

DETR is important because it changed object detection from a heavily engineered dense prediction pipeline into a set prediction architecture:

$$
\text{image features}
\rightarrow
\text{object queries}
\rightarrow
\text{matched prediction set}.
$$

This pattern influenced detection, segmentation, tracking, video, multimodal grounding, and general set-output architectures.

## Limitations

- Original DETR has slow convergence.
- Small-object performance is weaker without later improvements.
- Dense attention over image features can be expensive.
- Matching cost design is important.
- Learned queries are flexible but not always interpretable.
- Specialized detectors can still outperform it under practical constraints.

## Connections

- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/u-net|U-Net]]
- [[papers/architectures/index|Architecture papers]]
