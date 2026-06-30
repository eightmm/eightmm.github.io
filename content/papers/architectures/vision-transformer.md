---
title: An Image is Worth 16x16 Words
aliases:
  - papers/vision-transformer
  - papers/vit
tags:
  - papers
  - architectures
  - transformer
  - vision
---

# An Image is Worth 16x16 Words

> The paper showed that a standard Transformer encoder can be applied to image classification by treating fixed-size image patches as tokens.

## Metadata

| Field | Value |
| --- | --- |
| Paper | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale |
| Authors | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby |
| Year | 2020 preprint; 2021 conference |
| Venue | ICLR 2021 |
| arXiv | [2010.11929](https://arxiv.org/abs/2010.11929) |
| OpenReview | [YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy) |
| Status | verified |

## Question

Before ViT, attention was often combined with convolutional vision models or used inside otherwise convolutional architectures. The question was whether an image classifier could remove convolutional inductive bias almost entirely and rely on a Transformer over image patches.

The deeper architecture question is about representation units. CNNs treat images as dense grids and build locality, translation sharing, and hierarchy into the model. ViT asks whether an image can instead be treated like a sequence of visual tokens, with the Transformer learning the useful interactions from data.

This changes the vision modeling contract:

$$
\text{image grid}
\rightarrow
\text{patch sequence}
\rightarrow
\text{Transformer encoder}
$$

## Main Claim

With enough pretraining data, a pure Transformer encoder over patch tokens can match or exceed strong convolutional image classifiers after transfer.

Narrowed claim:

$$
X \in \mathbb{R}^{H \times W \times C}
\rightarrow
\{p_i\}_{i=1}^{N}
\rightarrow
\operatorname{TransformerEncoder}(\{e_i + \operatorname{pos}_i\})
$$

where each $p_i$ is a flattened image patch projected into a token embedding.

The important qualifier is "with enough pretraining data." ViT does not claim that patch Transformers are always better than CNNs in small-data settings.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor |
| Tokenization | split image into fixed-size non-overlapping patches |
| Token embedding | flatten each patch and linearly project to model dimension |
| Position signal | learned positional embedding |
| Backbone | Transformer encoder |
| Readout | class token representation |
| Natural task in paper | image classification and transfer |

For an image:

$$
X
\in
\mathbb{R}^{H \times W \times C}
$$

with patch size $P \times P$, the number of patches is:

$$
N
=
\frac{HW}{P^2}
$$

Each flattened patch has dimension:

$$
x_p^i
\in
\mathbb{R}^{P^2 C}
$$

and is projected into a token:

$$
e_i
=
x_p^i E
$$

where:

$$
E
\in
\mathbb{R}^{P^2 C \times D}
$$

The sequence length seen by the Transformer is $N+1$ because of the class token.

## Method

ViT splits the image into patches, linearly embeds each patch, adds positional embeddings, prepends a class token, and feeds the sequence to a Transformer encoder.

The patch embedding step is:

$$
z_0 =
[x_{\mathrm{class}};
x_p^1 E;
x_p^2 E;
\ldots;
x_p^N E]
+ E_{\mathrm{pos}}
$$

where $x_p^i$ is a flattened patch and $E$ is a learned projection.

The Transformer encoder then applies standard self-attention and feed-forward blocks:

$$
z_l
=
\operatorname{TransformerEncoderBlock}_l(z_{l-1})
$$

The image-level representation is taken from the final class token:

$$
h_{\text{cls}}
=
z_L^{0}
$$

and classification is:

$$
p(y \mid X)
=
\operatorname{softmax}(W h_{\text{cls}} + b)
$$

## Patch Embedding as Convolution

The patch projection can be viewed as a convolution with kernel size $P$, stride $P$, and output channels $D$.

$$
\text{linear patch projection}
\equiv
\text{Conv2D}(k=P, s=P, c_{\text{out}}=D)
$$

This view is useful because ViT is not completely free of image assumptions. It still chooses a patch grid and a fixed patch size. What it removes is the deep convolutional hierarchy inside the backbone.

| Design Choice | Consequence |
| --- | --- |
| large patch size | shorter sequence, less fine spatial detail |
| small patch size | longer sequence, higher attention cost |
| learned positional embeddings | captures training-resolution positions |
| class token | creates a sequence-level readout |
| pure Transformer encoder | weak local inductive bias compared with CNNs |

## Inductive Bias Tradeoff

CNNs build in locality and translation sharing:

$$
z_{i,j}
=
\sum_{u,v}
W_{u,v}X_{i+u,j+v}
$$

ViT uses global self-attention over patch tokens:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

This gives every patch a direct path to every other patch, but it does not force nearby pixels or patches to be treated as special.

| Model | Built-in Bias | Data Need |
| --- | --- | --- |
| CNN | locality, weight sharing, hierarchy | lower |
| ViT | global token mixing, weak image-specific bias | higher |
| Hybrid ViT | convolutional stem plus Transformer | middle |

The paper's central lesson is not "inductive bias is bad." It is that enough data and scale can compensate for weaker hand-designed vision bias.

## Complexity

Self-attention cost depends on the number of patches:

$$
O(N^2D)
$$

where:

$$
N
=
\frac{HW}{P^2}
$$

So patch size directly controls attention cost:

$$
P \uparrow
\Rightarrow
N \downarrow
\Rightarrow
\text{attention cost decreases}
$$

but larger patches may lose fine spatial information.

## Data Scale and Transfer

ViT is a data-scale paper as much as an architecture paper.

| Training Regime | Reading |
| --- | --- |
| small or mid-sized labeled data | CNN inductive bias can be stronger |
| large-scale supervised pretraining | ViT becomes competitive or better |
| transfer to downstream datasets | tests representation reuse rather than only ImageNet fitting |

The paper's architecture claim should therefore be tied to pretraining scale:

$$
\text{weak image-specific bias}
+
\text{large pretraining data}
\rightarrow
\text{strong transferable vision representation}
$$

This is the same broad pattern later seen in language and multimodal foundation models.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Pure Transformer vision models can perform strongly | image classification transfer results after large-scale pretraining | depends heavily on pretraining data scale |
| Patch tokenization is a viable image representation | comparison against strong CNN baselines | local inductive bias is weaker than CNNs |
| Data scale changes architecture ranking | smaller-data settings favor stronger inductive bias | not all domains have large pretraining data |

## Benchmark Reading

ViT evidence should be read through three axes.

| Axis | What to check |
| --- | --- |
| pretraining dataset | how large and how close to downstream evaluation |
| transfer protocol | whether the same backbone transfers cleanly |
| baseline strength | whether CNN baselines have comparable data and training recipe |

The headline result is not just ImageNet accuracy. It is that a generic Transformer encoder can become a high-quality vision backbone after large-scale pretraining.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | image classification |
| Input/output unit | image to class label |
| Main route | patch tokens to Transformer encoder |
| Main comparison | convolutional vision backbones and hybrid models |
| Not directly tested | dense segmentation as the core task, molecular graphs, protein structure |

## Ablation Reading

| Axis | What it tests | Reading |
| --- | --- | --- |
| patch size | token granularity and attention cost | architecture quality depends on representation unit |
| model size | whether Transformer scaling helps vision | larger models need enough data |
| pretraining dataset size | data-scale dependence | core to the paper's conclusion |
| hybrid vs pure ViT | value of convolutional image bias | CNN stems can help, especially under lower data |
| positional embeddings | spatial information injection | image token order is not inherent to attention |

The most important ablation lesson is that architecture and data scale interact. ViT's weakness under smaller data is not an implementation footnote; it is part of the paper's claim.

## Relation to Other Architecture Papers

| Paper | What Changes |
| --- | --- |
| [[papers/architectures/alexnet|AlexNet]] | makes deep CNNs practical at ImageNet scale |
| [[papers/architectures/deep-residual-learning|ResNet]] | makes very deep CNNs trainable |
| ViT | replaces convolutional backbone with patch-token Transformer |
| [[papers/architectures/emerging-properties-in-self-supervised-vision-transformers|DINO]] | shows self-supervised ViT features and attention maps can expose semantic object structure |
| [[papers/architectures/masked-autoencoders-are-scalable-vision-learners|MAE]] | pre-trains a ViT encoder through masked patch reconstruction |
| [[papers/architectures/swin-transformer|Swin Transformer]] | reintroduces hierarchy and local windows into vision Transformers |

ViT is the cleanest baseline for asking whether an image can be treated as a token sequence.

## Implementation Notes

- Patch embedding fixes the granularity of visual tokens; changing patch size changes sequence length and compute.
- Positional embeddings may need interpolation when transferring to different image resolutions.
- Strong augmentation and regularization matter; weak recipes can make ViT look worse than it is.
- The class token is a readout convention, not a law; later models use pooling or dense readouts.
- For dense prediction, plain ViT needs adaptation because it lacks a native feature pyramid.
- Data leakage or near-duplicate pretraining images can distort transfer claims.

## Limitations

- ViT trades convolutional locality for data-hungry global token mixing.
- The paper's headline strength depends on large-scale pretraining and transfer.
- Patch size, positional embedding, augmentation, regularization, and pretraining dataset all affect the architecture claim.
- Dense prediction and small-data vision require additional adaptations.
- Attention cost is quadratic in the number of patches.
- Patch tokenization can discard fine spatial detail when patches are too large.
- The paper does not prove that convolutional bias is unnecessary in all vision tasks.

## Why It Matters

ViT made Transformer encoders a general vision backbone and clarified when architectural inductive bias can be replaced by data scale.

The reusable pattern is:

$$
\text{structured input}
\rightarrow
\text{tokens}
\rightarrow
\text{Transformer encoder}
\rightarrow
\text{task head}
$$

This pattern matters beyond images. It is the same abstraction used when turning molecules, protein sequences, point clouds, or multimodal inputs into token-like units. The hard part is choosing tokens that preserve the right structure.

## Connections

- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/emerging-properties-in-self-supervised-vision-transformers|DINO]]
- [[papers/architectures/masked-autoencoders-are-scalable-vision-learners|MAE]]
- [[papers/architectures/swin-transformer|Swin Transformer]]
- [[papers/architectures/index|Architecture papers]]
