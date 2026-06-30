---
title: Group Normalization
aliases:
  - papers/group-normalization
  - papers/groupnorm
tags:
  - papers
  - architectures
  - normalization
  - computer-vision
---

# Group Normalization

> GroupNorm normalizes channels within each example, avoiding BatchNorm's dependence on mini-batch statistics while staying natural for convolutional feature maps.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Group Normalization |
| Authors | Yuxin Wu, Kaiming He |
| Year | 2018 |
| Venue | ECCV 2018 |
| arXiv | [1803.08494](https://arxiv.org/abs/1803.08494) |
| Paper | [CVF Open Access](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html) |
| Status | seed note started |

## One-Line Takeaway

GroupNorm keeps normalization inside each example by splitting channels into groups and computing statistics over channel-group plus spatial positions, making it stable when batch size is small or variable.

## Question

[[papers/architectures/batch-normalization|Batch Normalization]] works well in large-batch CNN training:

$$
\mu_c
=
\frac{1}{BHW}
\sum_{b,h,w}X_{bchw}.
$$

But detection, segmentation, and video models often use small batches because feature maps and models are large:

$$
B \in \{1,2,4\}.
$$

Batch statistics then become noisy:

$$
\hat{\mu}_B,\hat{\sigma}_B^2
\text{ are poor estimates of stable activation statistics}.
$$

GroupNorm asks:

> Can normalization keep CNN channel structure while removing dependence on the batch dimension?

## Architecture Contract

| Component | Role |
| --- | --- |
| channel groups | partition $C$ channels into $G$ groups |
| group statistics | compute mean and variance per example and group |
| affine scale/shift | restore per-channel representational flexibility |
| batch independence | same statistics axis at train and inference |
| CNN compatibility | preserve channel/spatial feature-map structure |

For input:

$$
X\in\mathbb{R}^{N\times C\times H\times W},
$$

GroupNorm partitions channels into $G$ groups. Each group has:

$$
C_g = C/G
$$

channels.

## Group Statistics

For example $n$ and group $g$, let:

$$
\mathcal{S}_{n,g}
=
\{(c,h,w): c\in g,\ 1\le h\le H,\ 1\le w\le W\}.
$$

The group mean is:

$$
\mu_{n,g}
=
\frac{1}{|\mathcal{S}_{n,g}|}
\sum_{(c,h,w)\in\mathcal{S}_{n,g}}
X_{nchw}.
$$

The group variance is:

$$
\sigma^2_{n,g}
=
\frac{1}{|\mathcal{S}_{n,g}|}
\sum_{(c,h,w)\in\mathcal{S}_{n,g}}
(X_{nchw}-\mu_{n,g})^2.
$$

For a channel $c$ inside group $g(c)$:

$$
Y_{nchw}
=
\gamma_c
\frac{
X_{nchw}-\mu_{n,g(c)}
}{
\sqrt{\sigma^2_{n,g(c)}+\epsilon}
}
+
\beta_c.
$$

The affine parameters remain per-channel:

$$
\gamma,\beta\in\mathbb{R}^{C}.
$$

## Axis Comparison

The architecture difference is the statistics axis.

| Method | Statistics Over | Depends On Batch? | Common Use |
| --- | --- | --- | --- |
| BatchNorm | batch and spatial axes per channel | yes | large-batch CNNs |
| LayerNorm | all channels/features within each example or token | no | sequence models, Transformers |
| InstanceNorm | spatial axes per example/channel | no | style transfer, image generation |
| GroupNorm | channel groups plus spatial axes per example | no | small-batch detection/segmentation/video |

GroupNorm sits between LayerNorm and InstanceNorm:

$$
G=1 \Rightarrow \text{LayerNorm-like over channels/spatial},
$$

$$
G=C \Rightarrow \text{InstanceNorm-like}.
$$

## Why It Matters

GroupNorm is a practical architecture component for vision models where BatchNorm's batch contract breaks.

| Situation | Why BatchNorm Struggles | Why GroupNorm Helps |
| --- | --- | --- |
| object detection | high-resolution features reduce batch size | statistics do not use batch axis |
| segmentation | dense feature maps are memory-heavy | train/eval formula stays consistent |
| video | temporal input increases memory | small batch does not corrupt statistics |
| fine-tuning | target batch differs from pretraining batch | no running batch-stat mismatch |

This makes GroupNorm important in architecture notes for [[concepts/tasks/object-detection|object detection]], [[concepts/tasks/segmentation|segmentation]], and high-memory vision models.

## What To Watch

- Group count $G$ is a hyperparameter and should be recorded.
- GroupNorm removes batch-stat fragility but does not automatically outperform BatchNorm in large-batch CNN training.
- It changes feature statistics and may require tuning learning rate, weight decay, and initialization.
- In distributed training, GroupNorm avoids cross-replica statistic synchronization that BatchNorm may need.
- The formula is train/eval consistent, but affine parameters still belong to model state.

## Related

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[papers/architectures/batch-normalization|Batch Normalization]]
- [[papers/architectures/layer-normalization|Layer Normalization]]
- [[papers/architectures/root-mean-square-layer-normalization|RMSNorm]]
- [[papers/architectures/mask-r-cnn|Mask R-CNN]]
