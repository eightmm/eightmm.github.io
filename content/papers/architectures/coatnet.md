---
title: CoAtNet
aliases:
  - papers/coatnet
  - papers/marrying-convolution-and-attention
tags:
  - papers
  - architectures
  - vision
  - cnn
  - attention
---

# CoAtNet

> The paper built a vision backbone family by combining convolutional inductive bias with attention capacity through a staged convolution-attention layout.

## Metadata

| Field | Value |
| --- | --- |
| Paper | CoAtNet: Marrying Convolution and Attention for All Data Sizes |
| Authors | Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan |
| Year | 2021 |
| Venue | NeurIPS 2021 |
| arXiv | [2106.04803](https://arxiv.org/abs/2106.04803) |
| Proceedings PDF | [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf) |
| Status | full note started |

## Question

Pure CNNs generalize well in limited-data image settings because locality, translation sharing, and hierarchy are built into the architecture. Pure Vision Transformers can scale to high capacity, but weak visual inductive bias often makes them data-hungry.

CoAtNet asks:

$$
\text{Can a vision backbone combine CNN generalization and Transformer capacity?}
$$

The paper's answer is a hybrid design:

$$
\text{early convolutional stages}
\rightarrow
\text{later attention stages}.
$$

This is a different axis from [[papers/architectures/mlp-mixer|MLP-Mixer]], which removes both convolution and attention. CoAtNet keeps both and asks how to arrange them.

## Main Claim

CoAtNet argues that convolution and attention should not be read as mutually exclusive vision backbones. A staged hybrid can use convolution where local inductive bias is most useful and attention where global capacity is most useful:

$$
\text{local bias at high resolution}
+
\text{global interaction at lower resolution}
\Rightarrow
\text{better generalization-capacity tradeoff}.
$$

The durable architecture claim is:

$$
\text{depthwise convolution and self-attention can be unified through relative attention,}
$$

and a vertical stack of convolution blocks followed by attention blocks is a strong practical backbone layout.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor |
| Backbone type | multi-stage hybrid ConvNet/Transformer vision backbone |
| Early stages | convolutional blocks, especially MBConv-style blocks |
| Later stages | relative self-attention blocks |
| Spatial schedule | progressive downsampling across stages |
| Main comparison | CNN generalization vs Transformer capacity |
| Natural task in paper | image classification under different data scales |

The high-level layout is:

$$
X_0
\xrightarrow{\text{stem}}
X_1
\xrightarrow{\text{conv stage}}
X_2
\xrightarrow{\text{conv stage}}
X_3
\xrightarrow{\text{attention stage}}
X_4
\xrightarrow{\text{attention stage}}
y.
$$

The exact stage pattern can vary, but the architectural idea is stable:

$$
\text{C-C-T-T}
\quad\text{or}\quad
\text{C-C-C-T}
$$

where $C$ denotes convolution-style blocks and $T$ denotes Transformer-style attention blocks.

## Why Convolution First

At high spatial resolution, dense attention is expensive:

$$
O(N^2d),
$$

where $N$ is the number of spatial tokens. Convolution at high resolution has a local cost:

$$
O(Nk^2d),
$$

where $k$ is the kernel size.

Early convolutional stages therefore do two things:

1. encode local visual patterns with strong image bias;
2. reduce spatial resolution before global attention is applied.

This is a practical answer to the same tension seen in [[papers/architectures/vision-transformer|Vision Transformer]] and [[papers/architectures/swin-transformer|Swin Transformer]]:

$$
\text{global attention is powerful, but raw high-resolution attention is expensive and data-hungry.}
$$

## Depthwise Convolution and Relative Attention

Depthwise convolution applies a local, channel-wise spatial filter:

$$
y_i
=
\sum_{j\in\mathcal{L}(i)}
w_{i-j}x_j,
$$

where $\mathcal{L}(i)$ is a local neighborhood and $w_{i-j}$ depends on relative offset.

Self-attention mixes values with input-dependent weights:

$$
y_i
=
\sum_j
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d}}
\right)
v_j.
$$

Relative attention adds a position-dependent term:

$$
y_i
=
\sum_j
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d}}
+
b_{i-j}
\right)
v_j.
$$

This makes the relationship clearer:

| Mechanism | Weight Depends On | Bias |
| --- | --- | --- |
| depthwise convolution | relative offset | fixed local spatial prior |
| self-attention | input content | adaptive global interaction |
| relative attention | input content and relative offset | adaptive interaction plus positional bias |

CoAtNet uses this relationship to justify a hybrid rather than treating convolution and attention as unrelated operations.

## MBConv as the Convolutional Block

CoAtNet uses the MobileNet/EfficientNet family as the convolutional side of the design. An MBConv-style block can be read as:

$$
x
\xrightarrow{\text{expand }1\times1}
h
\xrightarrow{\text{depthwise conv}}
u
\xrightarrow{\text{project }1\times1}
y.
$$

The inverted bottleneck pattern expands channels before spatial mixing and projects back for the residual path:

$$
d
\rightarrow
rd
\rightarrow
d.
$$

This matters because Transformer FFNs also use expansion and projection:

$$
d
\rightarrow
d_{\mathrm{ff}}
\rightarrow
d.
$$

CoAtNet's architecture comparison is therefore not only convolution vs attention. It also compares how similar expansion-projection blocks behave when the spatial mixing operator changes.

## Vertical Layout

The most useful reading of CoAtNet is the stage-ordering argument:

| Stage Position | Preferred Operator | Reason |
| --- | --- | --- |
| early, high resolution | convolution | cheap local bias and downsampling |
| middle | convolution or attention | transition depends on compute and data scale |
| late, lower resolution | attention | global interaction with fewer tokens |

This differs from simply adding attention everywhere. The paper's design principle is vertical composition:

$$
\text{Conv}
\rightarrow
\text{Conv}
\rightarrow
\text{Attention}
\rightarrow
\text{Attention}.
$$

The claim is that this ordering improves generalization, capacity, and efficiency together better than an ad hoc mixture.

## Relation to Nearby Papers

| Paper | What It Tests |
| --- | --- |
| [Vision Transformer](/papers/architectures/vision-transformer) | can a pure Transformer process image patches? |
| [MLP-Mixer](/papers/architectures/mlp-mixer) | are convolution and attention strictly necessary? |
| [Swin Transformer](/papers/architectures/swin-transformer) | can windowed hierarchical attention recover vision-friendly locality? |
| [ConvNeXt](/papers/architectures/convnext) | can a modernized pure ConvNet remain competitive after ViT-era design lessons? |
| [CoAtNet](/papers/architectures/coatnet) | can convolution and attention be staged to combine bias and capacity? |

Together, these papers define a useful vision backbone map:

$$
\text{CNN}
\leftrightarrow
\text{hybrid Conv-Attention}
\leftrightarrow
\text{pure attention}
\leftrightarrow
\text{all-MLP}.
$$

## Evidence to Read Carefully

The paper reports strong ImageNet results across different data regimes, including settings without extra data and larger pretraining settings. The headline numbers are useful, but the architecture lesson should be read through the ablation structure:

| Evidence Type | What It Supports |
| --- | --- |
| data-regime comparison | convolution helps when data is limited |
| scaling comparison | attention capacity helps at larger scale |
| stage-layout ablation | operator ordering matters |
| relative attention analysis | convolution and attention share a positional-mixing view |

The important check is whether the result comes from the hybrid block itself, the stage layout, model size, data scale, augmentation, or training recipe.

## Limits

- CoAtNet is a vision-specific backbone note, not a generic claim that every domain benefits from conv-attention hybrids.
- The design depends on image grids and progressive spatial downsampling.
- Strong reported performance is entangled with scale, training recipe, and resource budget.
- The paper is less conceptually minimal than ViT or MLP-Mixer because it combines several design choices.
- In domains without stable local grids, the convolutional side may not transfer naturally.

## What This Paper Teaches

CoAtNet is useful because it turns a binary architecture debate into an operator placement question:

$$
\text{Which operator should mix information at which resolution and data scale?}
$$

For architecture reading, keep these axes separate:

- local vs global interaction;
- fixed relative-offset bias vs content-dependent routing;
- high-resolution cost vs low-resolution capacity;
- data efficiency vs scaling capacity;
- block choice vs stage layout.

## Concepts

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]

## Related

- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/mlp-mixer|MLP-Mixer]]
- [[papers/architectures/swin-transformer|Swin Transformer]]
- [[papers/architectures/convnext|ConvNeXt]]
- [[papers/architectures/efficientnet|EfficientNet]]
- [[papers/architectures/mobilenetv2|MobileNetV2]]
