---
title: Masked Autoencoders Are Scalable Vision Learners
aliases:
  - papers/mae
  - papers/masked-autoencoder
  - papers/masked-autoencoders-are-scalable-vision-learners
tags:
  - papers
  - architectures
  - vision
  - transformer
  - self-supervised-learning
  - autoencoder
---

# Masked Autoencoders Are Scalable Vision Learners

> MAE makes masked image modeling efficient by encoding only visible image patches and using a lightweight decoder to reconstruct masked pixels.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Masked Autoencoders Are Scalable Vision Learners |
| Authors | Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick |
| Year | 2021 preprint; 2022 conference |
| Venue | CVPR 2022 |
| arXiv | [2111.06377](https://arxiv.org/abs/2111.06377) |
| Status | seed note |

## One-Line Takeaway

MAE turns ViT pre-training into a masked reconstruction problem with an asymmetric architecture:

$$
\text{image patches}
\rightarrow
\text{mask 75 percent}
\rightarrow
\text{encode visible patches only}
\rightarrow
\text{decode all patches}
\rightarrow
\text{reconstruct masked pixels}.
$$

The key architecture idea is not just masking. It is moving most compute to the visible subset.

## Question

Language models can learn from masked tokens because the model predicts missing symbols from context. MAE asks whether a similar masked reconstruction objective can scale for images.

For an image split into $N$ patches:

$$
X = \{x_1,\ldots,x_N\}.
$$

A random mask selects:

$$
\mathcal{M}\subset \{1,\ldots,N\},
\qquad
\mathcal{V}=\{1,\ldots,N\}\setminus \mathcal{M}.
$$

The self-supervised task is:

$$
\hat{x}_{\mathcal{M}}
=
f_\theta(x_{\mathcal{V}}),
$$

where only visible patches are observed.

The paper's deeper question is:

$$
\text{Can image SSL become efficient enough to train large ViT backbones?}
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image split into ViT-style patches |
| Pretext task | reconstruct masked image patches |
| Encoder input | visible patches only, no mask tokens |
| Encoder backbone | ViT encoder |
| Decoder input | encoded visible tokens plus mask tokens |
| Decoder role | lightweight reconstruction module |
| Mask ratio | high random masking, commonly around 75 percent |
| Training target | pixel reconstruction on masked patches |
| Downstream use | discard decoder, fine-tune or evaluate encoder |

MAE is both an architecture and a learning method. It belongs here because the visible-only encoder and asymmetric decoder are architecture decisions.

## Patch and Mask Setup

An image:

$$
I\in\mathbb{R}^{H\times W\times C}
$$

is split into patches:

$$
x_i\in\mathbb{R}^{P^2C},
\qquad
i=1,\ldots,N,
$$

where:

$$
N=\frac{HW}{P^2}.
$$

Each patch is embedded:

$$
e_i=x_iE,
\qquad
E\in\mathbb{R}^{P^2C\times D}.
$$

Then random masking keeps only visible patches:

$$
E_{\mathcal{V}}=\{e_i:i\in\mathcal{V}\}.
$$

If the mask ratio is $r$, then:

$$
|\mathcal{V}|=(1-r)N,
\qquad
|\mathcal{M}|=rN.
$$

For $r=0.75$, the encoder sees only:

$$
0.25N
$$

patch tokens.

## Asymmetric Encoder-Decoder

The encoder operates only on visible tokens:

$$
Z_{\mathcal{V}}
=
\operatorname{ViTEncoder}(E_{\mathcal{V}} + P_{\mathcal{V}}).
$$

Mask tokens are not passed through the encoder. This is the compute-saving move.

The decoder receives:

$$
[Z_{\mathcal{V}};\, m_{\mathcal{M}}] + P_{\text{dec}},
$$

where $m_{\mathcal{M}}$ are learned mask tokens placed at the missing patch positions.

Then:

$$
\hat{X}
=
\operatorname{Decoder}([Z_{\mathcal{V}};\, m_{\mathcal{M}}] + P_{\text{dec}}).
$$

The decoder predicts pixels for masked patches:

$$
\hat{x}_i
=
W_{\text{out}}\hat{z}_i,
\qquad
i\in\mathcal{M}.
$$

## Reconstruction Loss

The loss is computed only on masked patches:

$$
\mathcal{L}_{\text{MAE}}
=
\frac{1}{|\mathcal{M}|}
\sum_{i\in\mathcal{M}}
\left\lVert
\hat{x}_i - x_i
\right\rVert_2^2.
$$

This matters because reconstructing visible patches would make the task easier and less aligned with representation learning:

$$
\text{predict missing content}
\ne
\text{copy visible input}.
$$

The paper also discusses normalized pixel targets. When reading implementations, check whether patch pixels are normalized before reconstruction.

## Compute Logic

Dense ViT attention over all patches has rough attention cost:

$$
O(N^2D).
$$

MAE encoder attention sees only $(1-r)N$ patches:

$$
O(((1-r)N)^2D).
$$

For $r=0.75$:

$$
((1-r)N)^2=(0.25N)^2=0.0625N^2.
$$

So the expensive encoder attention is much smaller than full-patch pre-training.

The decoder sees all positions, but it is intentionally lightweight:

| Component | Token Count | Capacity | Purpose |
| --- | --- | --- | --- |
| encoder | visible patches only | large ViT backbone | learn representation |
| decoder | visible latents plus mask tokens | lightweight | reconstruct pixels |

This is why MAE is not just a denoising autoencoder copied into vision. Its asymmetric compute allocation is central.

## Why High Mask Ratio Works

Images are spatially redundant. A low mask ratio can make reconstruction too easy:

$$
\text{nearby visible patches}
\rightarrow
\text{local interpolation}.
$$

High masking forces the encoder to use broader context:

$$
\text{few visible patches}
\rightarrow
\text{global structure and semantics}.
$$

The paper reports that a high mask ratio such as 75 percent is effective for image MAE pre-training.

## Relation to ViT

| Paper | Role |
| --- | --- |
| [Vision Transformer](/papers/architectures/vision-transformer) | turns images into patch tokens and uses a Transformer encoder |
| MAE | pre-trains a ViT encoder by masked patch reconstruction |

MAE inherits the ViT tokenization contract:

$$
\text{image}
\rightarrow
\text{patch sequence}
\rightarrow
\text{Transformer encoder}.
$$

It changes the training route:

$$
\text{supervised image labels}
\rightarrow
\text{self-supervised masked reconstruction}.
$$

## Relation to BERT-Style Masking

MAE is inspired by masked modeling, but image and text differ:

| Axis | BERT-style text masking | MAE image masking |
| --- | --- | --- |
| unit | token IDs | image patches |
| target | discrete token prediction | pixel reconstruction |
| redundancy | lower local redundancy | high local spatial redundancy |
| mask ratio | moderate | high |
| encoder input | often includes mask token | visible patches only |

The visible-only encoder is the major difference from simply putting `[MASK]` patches into a ViT.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| asymmetric encoder-decoder is efficient | runtime and training comparisons | visible-only encoding reduces compute | wall-clock depends on implementation and hardware |
| high mask ratio works well | mask-ratio ablations | image MAE benefits from difficult reconstruction | optimal ratio can depend on patch size and domain |
| representation transfers | ImageNet fine-tuning and downstream transfer | encoder learns useful visual features | downstream performance also depends on fine-tuning recipe |
| large ViTs scale under MAE | larger-backbone experiments | SSL can train high-capacity vision models | compute and dataset assumptions still matter |

## Implementation Reading

Check:

- patch size and image resolution;
- random masking policy and mask ratio;
- whether the encoder receives mask tokens;
- encoder depth, decoder depth, decoder width;
- whether reconstruction uses raw or normalized pixels;
- whether loss is computed only over masked patches;
- whether the decoder is discarded for downstream evaluation;
- whether results are linear probe, fine-tuning, or transfer;
- whether data augmentation and fine-tuning recipes are comparable to supervised baselines.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "MAE is just BERT for images." | It uses a visible-only encoder and lightweight decoder because images have different redundancy and reconstruction targets. |
| "The decoder is the main model." | The decoder is mainly a pre-training helper; the encoder is the transferred backbone. |
| "More mask tokens make the encoder harder." | In MAE, mask tokens are kept out of the encoder. |
| "Pixel reconstruction means low-level features only." | The high mask ratio and transfer evidence argue that useful representations can emerge, but this must be checked downstream. |
| "MAE is only a learning objective." | The asymmetric encoder-decoder is an architecture decision. |

## What to Remember

MAE belongs in the architecture shelf because it changes where compute is spent:

$$
\text{large encoder on visible patches}
+ \text{small decoder on all patches}
\rightarrow
\text{scalable masked image pre-training}.
$$

The general lesson:

$$
\text{pretext task difficulty}
+ \text{architecture asymmetry}
+ \text{compute allocation}
=
\text{scalable SSL}.
$$

This is useful beyond natural images whenever the input can be split into parts and a model can learn from partially observed structure.

## Links

- [[concepts/architectures/autoencoder|Autoencoder]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/index|Architecture papers]]
