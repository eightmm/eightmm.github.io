---
title: Segment Anything
aliases:
  - papers/segment-anything
  - papers/sam
  - papers/segment-anything-model
tags:
  - papers
  - architectures
  - computer-vision
  - segmentation
  - foundation-model
  - promptable-model
  - transformer
---

# Segment Anything

> The paper turns segmentation into a promptable foundation-model interface: encode the image once, encode prompts cheaply, and decode masks interactively.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Segment Anything |
| Authors | Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick |
| Year | 2023 |
| Venue | ICCV 2023 |
| arXiv | [2304.02643](https://arxiv.org/abs/2304.02643) |
| Paper | [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html) |
| Status | seed note started |

## One-Line Takeaway

SAM separates image encoding, prompt encoding, and mask decoding so the same image representation can answer many segmentation prompts without retraining on each downstream segmentation dataset.

## Question

Classical segmentation models are often trained for one label space or one dataset:

$$
f_\theta(I)
\rightarrow
\hat{Y}
\in
\{1,\ldots,C\}^{H\times W}.
$$

SAM changes the interface:

$$
f_\theta(I, p)
\rightarrow
\{\hat{m}_1,\hat{m}_2,\hat{m}_3\},
$$

where $p$ is a prompt such as:

- point;
- box;
- coarse mask;
- text in broader promptable segmentation framing, although the released SAM model focuses on visual prompts.

The model asks:

> Can segmentation be treated as a general promptable task rather than a fixed closed-set prediction problem?

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| image encoder | image $I$ | dense image embedding $E_I$ | expensive reusable representation |
| prompt encoder | points, boxes, masks | sparse and dense prompt embeddings $E_p$ | condition segmentation on user/task intent |
| mask decoder | $E_I, E_p$ | masks and quality scores | produce candidate masks for the prompt |

The interface is:

$$
E_I = \operatorname{ImageEncoder}(I),
$$

$$
E_p = \operatorname{PromptEncoder}(p),
$$

$$
(\hat{m}_{1:K}, \hat{q}_{1:K})
=
\operatorname{MaskDecoder}(E_I, E_p).
$$

Here $\hat{q}_k$ estimates mask quality for candidate mask $\hat{m}_k$.

## Image Encoder

The image encoder is the heavy backbone. It maps the image into a spatial embedding:

$$
I \in \mathbb{R}^{3\times H\times W}
\rightarrow
E_I \in \mathbb{R}^{C\times H'\times W'}.
$$

This is the expensive part, so SAM's system design encourages reuse:

$$
\text{encode image once}
\rightarrow
\text{answer many prompts}.
$$

That is an architecture decision, not just an implementation detail. Interactive segmentation requires low prompt-to-mask latency.

## Prompt Encoder

Prompt inputs are heterogeneous:

| Prompt | Representation Need |
| --- | --- |
| point | coordinate plus positive/negative label |
| box | two corner coordinates |
| mask | dense spatial prior |

The prompt encoder maps them into embeddings that can condition the decoder:

$$
p
\rightarrow
E_p.
$$

The important abstraction:

$$
\text{user/task intent}
\rightarrow
\text{conditioning tokens}.
$$

This is why SAM belongs in an architecture shelf, not only a segmentation task page. It changes the model's input contract.

## Mask Decoder

The mask decoder combines image and prompt information:

$$
\operatorname{Decoder}(E_I, E_p)
\rightarrow
(\hat{m}, \hat{q}).
$$

A simplified attention view is:

$$
\operatorname{CrossAttn}(Q_p, K_I, V_I)
=
\operatorname{softmax}
\left(
\frac{Q_pK_I^\top}{\sqrt{d}}
\right)V_I,
$$

where:

- $Q_p$ comes from prompt or output tokens;
- $K_I,V_I$ come from image embeddings;
- the decoder uses the prompt to retrieve relevant image regions.

SAM produces multiple masks for ambiguous prompts:

$$
\{\hat{m}_1,\hat{m}_2,\hat{m}_3\}
=
f_\theta(I,p).
$$

This matters because a single point can correspond to several valid masks, such as part, object, or group.

## Promptable Segmentation vs Instance Segmentation

[[papers/architectures/mask-r-cnn|Mask R-CNN]] predicts masks for proposed object instances:

$$
I
\rightarrow
\{(\hat{c}_i,\hat{b}_i,\hat{m}_i)\}.
$$

SAM predicts masks conditioned by prompts:

$$
(I,p)
\rightarrow
\{\hat{m}_k\}.
$$

| Axis | Mask R-CNN | SAM |
| --- | --- | --- |
| Object proposal | RPN/RoI pipeline | user or algorithmic prompt |
| Output | class, box, mask | mask candidates and quality |
| Class labels | explicit categories | class-agnostic masks |
| Main interface | detector | promptable segmenter |
| Transfer route | supervised task adaptation | zero-shot prompt transfer |

## Why It Matters

SAM is a foundation-model architecture pattern for vision:

$$
\text{large reusable encoder}
+
\text{small prompt interface}
+
\text{fast decoder}.
$$

The central lesson is that architecture can define a reusable user-facing interface, not only a better benchmark model. This connects SAM to broader patterns in [[ai/architectures|Architectures]], [[agents/tools/tool-use|Tool Use]], and interactive annotation systems.

## What To Watch

- SAM is not a semantic classifier; it returns masks, not category labels.
- Strong zero-shot behavior depends on the model, data engine, and SA-1B scale together.
- Prompt choice changes the output; evaluation must specify prompt protocol.
- Domain-specific images can fail when visual priors differ from natural-image training data.
- Mask quality and downstream usefulness are different claims.

## Related

- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/modalities/image|Image]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/mask-r-cnn|Mask R-CNN]]
- [[papers/architectures/detr|DETR]]
