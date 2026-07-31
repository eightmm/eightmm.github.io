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

## Transformer Encoder Contract

The patch sequence is processed by a standard pre-normalization Transformer encoder. For one block, let

$$
Z\in\mathbb{R}^{(N+1)\times D}.
$$

The self-attention sublayer is:

$$
Q=Z W_Q,
\qquad
K=Z W_K,
\qquad
V=Z W_V,
$$

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),
$$

$$
\operatorname{MSA}(Z)=A V W_O.
$$

For $M$ heads, the head outputs are concatenated before the output projection:

$$
\operatorname{MSA}(Z)
=
\operatorname{Concat}(A_1V_1,\ldots,A_MV_M)W_O.
$$

The feed-forward sublayer is applied independently to each token:

$$
\operatorname{MLP}(Z)
=
\phi(ZW_1+b_1)W_2+b_2.
$$

With pre-normalization and residual paths, one block is:

$$
U=Z+\operatorname{MSA}(\operatorname{LN}(Z)),
$$

$$
Z'=U+\operatorname{MLP}(\operatorname{LN}(U)).
$$

This is the reusable architecture, while the image-specific part is the conversion of $X$ into $Z_0$. A paper note should keep these two contracts separate. Otherwise, later changes to patch embedding or positional encoding can be mistaken for changes to attention itself.

## Multi-Head Shape Trace

Let the model width be $D$ and the per-head width be $d_k=D/M$. For one input sequence:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| $Z$ | $(N+1)\times D$ | class token plus patch tokens |
| $Q,K,V$ per head | $(N+1)\times d_k$ | projected token features |
| $QK^\top$ | $(N+1)\times(N+1)$ | pairwise token scores |
| $A$ | $(N+1)\times(N+1)$ | row-normalized attention weights |
| head output | $(N+1)\times d_k$ | weighted value mixture |
| concatenated output | $(N+1)\times D$ | all head channels |
| MLP hidden | $(N+1)\times D_{\mathrm{mlp}}$ | per-token channel expansion |

The class token participates in the same attention operation as patch tokens. Its final representation is used as the global readout in the canonical setup:

$$
h_{\mathrm{image}}=Z_L[0,:].
$$

This means the class token is not a separate pooling operation outside the Transformer. It is a learned query-like participant in every layer and can gather information from all patches.

## Parameter and FLOP Decomposition

Ignoring biases, one attention block with model width $D$ and MLP expansion width $D_{\mathrm{mlp}}$ has approximate parameter count:

$$
P_{\mathrm{attn}}
\approx
4D^2,
$$

because the $Q,K,V$ projections and output projection each have roughly $D^2$ parameters. The MLP contributes:

$$
P_{\mathrm{mlp}}
\approx
2D D_{\mathrm{mlp}}.
$$

For $N+1$ tokens, the dominant attention computation is approximately:

$$
\operatorname{FLOPs}_{\mathrm{attn}}
\propto
4(N+1)D^2
+
2(N+1)^2D,
$$

where the first term represents projections and the second represents score/value interactions. The MLP computation is approximately:

$$
\operatorname{FLOPs}_{\mathrm{mlp}}
\propto
2(N+1)D D_{\mathrm{mlp}}.
$$

Since

$$
N=\frac{HW}{P^2},
$$

the patch size changes both the input representation and the quadratic attention term. Halving $P$ in each spatial direction multiplies $N$ by four and can multiply the pairwise term by sixteen.

| Change | Sequence effect | Main cost effect |
| --- | --- | --- |
| increase image resolution | $N$ grows quadratically with image side | attention pair cost grows sharply |
| decrease patch size | more tokens preserve finer detail | same quadratic penalty |
| increase model width $D$ | token representation becomes richer | projection and MLP costs grow with $D^2$ |
| increase depth $L$ | repeated refinement | cost roughly scales with number of blocks |

This is why later hierarchical models such as [[papers/architectures/swin-transformer|Swin Transformer]] alter the interaction pattern rather than simply making global ViT larger.

## Patch Size Is an Information and Systems Choice

Patch size is often described as a hyperparameter, but it defines the basic visual unit:

$$
\text{pixel grid}
\xrightarrow{P\times P\ \text{grouping}}
\text{token grid}.
$$

For a fixed image size, smaller patches provide more tokens and a finer spatial lattice. Larger patches reduce the sequence length but force the first projection to summarize more local variation.

| Patch choice | Benefit | Risk |
| --- | --- | --- |
| large $P$ | lower attention and memory cost | small objects and boundaries are compressed early |
| small $P$ | finer spatial representation | quadratic token interaction and larger activation memory |
| variable patch size | possible multi-scale efficiency | positional and batching contract becomes more complex |

The first patch projection is also where the model commits to a stride. There is no later operation that can recover spatial information that was discarded by an overly coarse non-overlapping patchification.

## Positional Embedding Contract

Self-attention by itself is permutation-equivariant over the token sequence. If the patch tokens are permuted and no position signal changes, the attention computation has no way to distinguish their spatial locations. ViT therefore adds a learned position matrix:

$$
Z_0
=
[x_{\mathrm{class}};X_p]E
+E_{\mathrm{pos}},
$$

where

$$
E_{\mathrm{pos}}
\in
\mathbb{R}^{(N+1)\times D}.
$$

The class-token row and patch-token rows have different semantic roles. When transferring to another resolution, the patch-token grid changes from $H/P\times W/P$ to a new grid. A common adaptation is to interpolate the patch portion of $E_{\mathrm{pos}}$ while preserving the class-token row.

The interpolation is an engineering operation, not a new learned architecture claim. Record it in the transfer protocol because a checkpoint used at a new resolution is not identical to the original training setup.

## Class Token Versus Global Average Pooling

The canonical ViT reads the class token, but the same encoder can expose a pooled representation:

$$
h_{\mathrm{gap}}
=
\frac1N\sum_{i=1}^{N}Z_L[i,:].
$$

The choice changes the readout contract without changing the internal attention blocks.

| Readout | Interpretation | Evaluation question |
| --- | --- | --- |
| class token | learned global collector | does a dedicated token gather useful evidence? |
| global average pooling | symmetric aggregation of patch features | does the patch representation already contain enough global information? |
| dense patch features | preserve spatial outputs | can the backbone support localization or segmentation? |

When comparing two ViT implementations, do not call them the same model if one changes the readout, fine-tuning head, or pooling normalization without recording it.

## Model Variants and Scaling

ViT is a family, not one fixed model. A variant is determined by at least:

$$
\mathcal{V}
=
(P,D,L,M,D_{\mathrm{mlp}},N,\text{recipe},\text{pretraining data}).
$$

Here $P$ is patch size, $D$ model width, $L$ depth, $M$ number of heads, $D_{\mathrm{mlp}}$ feed-forward width, and $N$ sequence length. Two experiments both labeled “ViT-B” can still differ in image resolution, patch size, pretraining data, or fine-tuning recipe.

| Scaling axis | What increases | What it tests |
| --- | --- | --- |
| width | $D$ and head dimensions | feature capacity and matrix efficiency |
| depth | number of encoder blocks | iterative token interaction and abstraction |
| resolution | $H,W$ and therefore $N$ | fine detail versus quadratic cost |
| patch granularity | $P$ and $N$ | input compression and spatial fidelity |
| pretraining data | examples and diversity | whether weak image bias can be learned from data |

The paper's central comparison is therefore a joint scaling statement. A small-data, low-resolution, shallow ViT is not an adequate proxy for the regime in which the paper's main claim was established.

## What the Paper Actually Compares

The strongest interpretation of the paper's evidence is:

$$
\text{pure patch Transformer}
+
\text{large-scale pretraining}
\rightarrow
\text{strong transferable visual representation}.
$$

It does not isolate every possible difference between CNNs and Transformers. The comparison includes choices about model scale, pretraining data, compute, augmentation, and transfer protocol.

| Comparison question | Evidence needed |
| --- | --- |
| Can patch tokens support image classification? | matched classification evaluation |
| Does ViT transfer to smaller datasets? | same downstream protocol and data budget |
| Is the result due to attention rather than scale? | parameter/FLOP/data-matched controls |
| Does ViT provide a general vision backbone? | dense prediction or localization evaluation |
| Does patchification preserve small objects? | resolution and object-size stratification |

This boundary matters when using ViT as a baseline for scientific images. A result on image classification should not automatically be described as a result on dense spatial reasoning.

## Training Recipe and Data Regime

ViT's apparent data hunger is part of its architectural story. A CNN starts with locality and weight sharing; a patch Transformer must learn more of the useful spatial structure from examples. The comparison can be expressed as:

$$
\text{effective inductive bias}
=
\text{architecture prior}
+
\text{data and objective signal}.
$$

For a fair reproduction, capture:

| Variable | Why it matters |
| --- | --- |
| pretraining dataset size | compensates for weaker hand-designed locality |
| label quality and duplicates | changes the amount of usable supervision |
| optimizer and schedule | affects large Transformer convergence |
| augmentation and regularization | affects small-data generalization |
| weight decay and dropout | changes the effective capacity |
| fine-tuning resolution | changes positional embedding and token count |
| initialization and warmup | can affect stability at scale |

Comparing ViT with a CNN trained using a stronger or weaker recipe can answer a recipe question rather than an architecture question.

## Transfer and Resolution Changes

Suppose a checkpoint was trained with patch grid $G_0$ and is transferred to grid $G_1$. The token projection and Transformer width may remain unchanged, while the positional table must be adapted:

$$
E_{\mathrm{pos}}^{(G_0)}
\rightarrow
\widetilde E_{\mathrm{pos}}^{(G_1)}.
$$

The transfer contract should state:

1. original image resolution and patch size;
2. target image resolution and patch size;
3. whether patch size is unchanged;
4. positional interpolation method;
5. whether the classification head is replaced;
6. whether all layers or only the head are fine-tuned.

If $P$ changes, the first projection itself no longer has the same input shape $P^2C$. That is a different adaptation problem from changing only the image resolution.

## Dense Prediction Boundary

The original ViT setup is naturally a classification model:

$$
X\rightarrow Z_L[0,:]\rightarrow y.
$$

Dense prediction requires patch-level or multi-scale features:

$$
X\rightarrow Z_L[1:N+1,:]
\rightarrow
\text{decoder or feature pyramid}
\rightarrow
Y_{\mathrm{dense}}.
$$

The vanilla architecture has no native hierarchy of resolutions. Later designs address this in different ways:

| Design | Added structure |
| --- | --- |
| hierarchical Transformer | progressively merges tokens and restricts interaction windows |
| decoder bridge | reshapes tokens and upsamples them for dense outputs |
| feature pyramid adapter | projects selected layers into multi-scale maps |
| hybrid stem | adds convolutional locality before global token mixing |

Therefore, a ViT classification result and a ViT detection result should not be treated as the same architecture contract.

## Ablation Matrix

| Ablation | Hold fixed | Main question |
| --- | --- | --- |
| patch size | data and model budget | how much spatial detail is lost or gained? |
| class token versus pooling | encoder and recipe | which global readout is more robust? |
| positional embedding | patch sequence and encoder | how much does explicit spatial identity matter? |
| pretraining scale | architecture | can data substitute for convolutional bias? |
| model width/depth | data and recipe | where does ViT scaling saturate? |
| hybrid stem | token count and encoder | does early local processing improve sample efficiency? |
| fine-tuning resolution | checkpoint and downstream data | how robust is position adaptation? |
| regularization | architecture and data | is the observed gap optimization or representation? |

For each row, check whether the change also alters parameters, FLOPs, sequence length, or activation memory. A patch-size ablation is rarely a pure representation ablation because it changes the attention budget at the same time.

## Failure Modes

- **Small-data overclaim:** a ViT trained from scratch on a small dataset is not a direct test of the paper's large-pretraining regime.
- **Patch aliasing:** fixed non-overlapping patches can discard boundaries, textures, or small objects before the first Transformer layer.
- **Quadratic scaling:** global attention becomes expensive as resolution or patch density grows.
- **Position mismatch:** using learned positional embeddings at a new grid without a documented adaptation can invalidate transfer comparisons.
- **Backbone/head confusion:** a dense-prediction result may owe substantial performance to the decoder or feature pyramid.
- **Recipe mismatch:** optimizer, augmentation, regularization, and pretraining data can dominate the comparison with CNNs.
- **Representation shortcut:** global token mixing can exploit dataset-specific correlations that do not transfer under distribution shift.

## Transfer to Scientific and Biological Data

ViT is useful for any domain where a structured object can be converted into meaningful patches, but patchification must preserve the task's symmetries and resolution requirements.

| Scientific input | ViT-style tokenization | Main question |
| --- | --- | --- |
| microscopy image | non-overlapping image patches | are cellular boundaries smaller than the patch? |
| protein contact/distance map | square map patches | does patch order preserve pairwise residue semantics? |
| voxelized molecular field | 3D patch or projection tokens | how are rotations and translations handled? |
| molecular surface raster | local surface patches | is orientation encoded or normalized? |
| protein sequence | sequence tokens, not image patches | should sequence order and residue identity be modeled directly? |
| molecular graph | graph tokens or graph-to-set adapter | does a grid tokenization destroy adjacency? |

For coordinate-based molecular inputs, vanilla ViT is not translation- or rotation-equivariant by construction. If the target should respect those transformations, compare it with a geometric architecture such as [[papers/architectures/egnn|EGNN]] or a suitable SE(3)-equivariant model. A ViT baseline can still be informative, but its symmetry assumptions must be stated.

## Reproduction Specification

Before reproducing or adapting ViT, record:

| Field | Required value |
| --- | --- |
| image shape | $H\times W\times C$ |
| patch shape | $P\times P$ and stride |
| token count | $N=HW/P^2$ plus class token policy |
| model width | $D$ |
| depth | number of encoder blocks $L$ |
| heads | $M$ and per-head width |
| MLP width | $D_{\mathrm{mlp}}$ and activation |
| normalization | axes, placement, epsilon |
| position signal | learned, fixed, relative, or interpolated |
| readout | class token, pooling, or dense features |
| pretraining | dataset, objective, duration, and checkpoint |
| transfer | target resolution, head, and fine-tuning policy |
| system | precision, batch size, memory optimization, and hardware |

Minimum shape checks:

1. Patchification produces exactly $N=HW/P^2$ patches.
2. The patch projection produces width $D$.
3. The class token and positional table have compatible sequence length.
4. Each attention head has the declared $d_k$.
5. The residual stream preserves $(N+1)\times D$ through every block.
6. The classifier reads the documented representation.
7. Resolution transfer changes only the declared components.

## One-Line Comparison With ConvNeXt

The cleanest comparison is:

$$
\begin{aligned}
\text{ViT:}&\quad \text{patch tokens}+\text{global content-dependent mixing}\\
\text{ConvNeXt:}&\quad \text{grid features}+\text{local depthwise mixing}.
\end{aligned}
$$

Both can use residual blocks, LayerNorm-like normalization, GELU, large-scale pretraining, and modern regularization. The difference that remains is the spatial mixing operator and the resulting inductive bias.

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
