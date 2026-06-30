---
title: MetaFormer
aliases:
  - papers/metaformer
  - papers/poolformer
  - papers/metaformer-is-actually-what-you-need-for-vision
tags:
  - papers
  - architectures
  - vision
  - transformer
---

# MetaFormer

> The paper argues that the general Transformer-style scaffold matters more than the specific attention token mixer for many vision backbones.

## Metadata

| Field | Value |
| --- | --- |
| Paper | MetaFormer Is Actually What You Need for Vision |
| Authors | Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, Shuicheng Yan |
| Year | 2022 |
| Venue | CVPR 2022 |
| arXiv | [2111.11418](https://arxiv.org/abs/2111.11418) |
| CVF | [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.html) |
| Code | [sail-sg/poolformer](https://github.com/sail-sg/poolformer) |
| Status | full note started |

## Question

Vision Transformers made attention the default explanation for why Transformer-like models work in vision. MLP-like models then showed that attention can be replaced by spatial MLPs while retaining competitive behavior.

MetaFormer asks a sharper question:

$$
\text{Is attention the key, or is the Transformer-style block scaffold the key?}
$$

The paper deliberately replaces attention with an extremely simple pooling operator. The resulting model, PoolFormer, still performs competitively on vision benchmarks. This supports the paper's hypothesis:

$$
\text{MetaFormer scaffold}
>
\text{specific token mixer}
$$

for a broad class of vision backbones.

## Main Claim

The paper abstracts a family of recent vision models into a common block:

$$
\text{MetaFormer block}
=
\text{Norm}
\rightarrow
\text{Token Mixer}
\rightarrow
\text{Residual}
\rightarrow
\text{Norm}
\rightarrow
\text{Channel MLP}
\rightarrow
\text{Residual}.
$$

The token mixer can be attention, spatial MLP, convolution, pooling, or another operator. The claim is not that token mixers do not matter. The claim is that the scaffold itself is a strong architecture prior:

$$
\text{token mixer choice}
\neq
\text{entire architecture explanation}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image features or patch tokens |
| Backbone type | hierarchical MetaFormer-style vision backbone |
| Token mixer | unspecified in the abstraction; pooling in PoolFormer |
| Channel mixer | MLP / feed-forward block |
| Normalization | normalization before mixer blocks |
| Residual path | residual connections around token mixer and channel MLP |
| Natural task in paper | image classification, detection, segmentation evidence |

The abstract block can be written as:

$$
U = X + \operatorname{TokenMixer}(\operatorname{Norm}_1(X)),
$$

$$
Y = U + \operatorname{MLP}(\operatorname{Norm}_2(U)).
$$

This is close to the standard Transformer block, except the token mixer is left open:

$$
\operatorname{TokenMixer}
\in
\{\text{attention}, \text{pooling}, \text{MLP}, \text{convolution}, \ldots\}.
$$

## PoolFormer as the Test Case

PoolFormer is the deliberately simple instantiation:

$$
\operatorname{TokenMixer}(X)
=
\operatorname{Pool}(X) - X.
$$

For a spatial feature map $X$, average pooling computes local neighborhood statistics:

$$
\operatorname{Pool}(X)_{u,v,c}
=
\frac{1}{|\mathcal{N}(u,v)|}
\sum_{(i,j)\in\mathcal{N}(u,v)}
X_{i,j,c}.
$$

The subtraction keeps the mixer focused on local context difference:

$$
\operatorname{PoolMixer}(X)
=
\operatorname{AvgPool}(X)-X.
$$

There are no attention scores, learned convolution kernels, or token-axis MLP weights in this mixer. That makes PoolFormer a controlled architecture probe:

$$
\text{if simple pooling works well, the scaffold deserves credit.}
$$

## MetaFormer vs Transformer

| Component | Transformer | MetaFormer |
| --- | --- | --- |
| token mixer | self-attention | any token-mixing operator |
| channel mixer | feed-forward network | MLP / feed-forward network |
| normalization | usually pre-norm or post-norm | normalization around both mixer stages |
| residual structure | residual around attention and FFN | residual around token mixer and MLP |
| claim | attention-based token mixing is central | scaffold may be the central reusable form |

This reframes [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]] for vision. The original Transformer made attention the dominant sequence mixer. MetaFormer asks whether the architecture skeleton around the mixer is also doing much of the work.

## Relation to Nearby Vision Backbones

| Paper | Token Mixer | What MetaFormer Helps Ask |
| --- | --- | --- |
| [Vision Transformer](/papers/architectures/vision-transformer) | global self-attention | is global attention necessary or just one mixer? |
| [MLP-Mixer](/papers/architectures/mlp-mixer) | token-axis MLP | can MLP token mixing replace attention? |
| [Swin Transformer](/papers/architectures/swin-transformer) | shifted-window attention | what happens when token mixing is local and hierarchical? |
| [CoAtNet](/papers/architectures/coatnet) | convolution then relative attention | where should different token mixers appear by stage? |
| [ConvNeXt](/papers/architectures/convnext) | depthwise convolution | can modern ConvNets be read as MetaFormer-like blocks? |
| [MetaFormer](/papers/architectures/metaformer) | abstract token mixer; pooling in PoolFormer | how much performance comes from the scaffold? |

This is why MetaFormer belongs in architecture papers. It is less about one model family and more about the decomposition of a modern backbone.

## Token Mixer Is a Slot

The key abstraction is to treat token mixing as a replaceable slot:

$$
\operatorname{TokenMixer}: \mathbb{R}^{N\times C}\rightarrow\mathbb{R}^{N\times C}.
$$

Examples:

| Mixer | Operation | Bias |
| --- | --- | --- |
| self-attention | content-dependent weighted sum | adaptive global interaction |
| window attention | content-dependent local/window sum | local adaptive interaction |
| token MLP | learned fixed token-axis mixing | learned position interaction |
| depthwise convolution | local learned kernel | locality and translation sharing |
| pooling | local non-parametric average | simple local smoothing |
| identity | no token mixing | tests scaffold lower bound |

For reading architecture papers, this slot view prevents over-attributing results to a named mechanism. A result may come from the mixer, but it may also come from normalization, residual layout, MLP ratio, hierarchical downsampling, training recipe, or regularization.

## Evidence to Read Carefully

The paper uses PoolFormer to test whether a very simple token mixer can still produce competitive vision models. It also compares against Transformer-like and MLP-like baselines.

Read the evidence along these axes:

| Evidence | What It Supports |
| --- | --- |
| PoolFormer classification results | simple token mixing can be enough inside the scaffold |
| downstream task transfer | the scaffold is not only an ImageNet classifier trick |
| comparisons with DeiT and ResMLP | attention and MLP token mixers are not the only viable choices |
| ablation of block design | normalization/residual/channel MLP are part of the claim |

The correct takeaway is:

$$
\text{architecture scaffold is an active hypothesis, not neutral boilerplate.}
$$

## Limits

- Pooling as a token mixer is mainly a proof device, not necessarily the best final design.
- Strong performance still depends on training recipe, stage design, resolution, and data.
- The claim is most directly about vision backbones, not every Transformer use case.
- It does not prove attention is unimportant; it shows attention is not the only explanation.
- Later MetaFormer variants can blur the line between concept paper and performance recipe.

## What This Paper Teaches

MetaFormer is useful because it gives a compact checklist for architecture reading:

$$
\text{model}
=
\text{tokenization}
+
\text{stage layout}
+
\text{token mixer}
+
\text{channel mixer}
+
\text{normalization}
+
\text{residual path}
+
\text{training recipe}.
$$

When a paper claims a new mixer is better, ask:

- Is the scaffold fixed?
- Is the MLP ratio fixed?
- Is the normalization placement fixed?
- Is the stage layout fixed?
- Are parameter count, FLOPs, data, augmentation, and training length matched?
- Does the mixer improve all tasks or only the benchmark shown?

## Concepts

- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/architecture-selection|Architecture selection]]

## Related

- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/mlp-mixer|MLP-Mixer]]
- [[papers/architectures/swin-transformer|Swin Transformer]]
- [[papers/architectures/coatnet|CoAtNet]]
- [[papers/architectures/convnext|ConvNeXt]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
