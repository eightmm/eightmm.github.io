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
| Status | full paper note |

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

## Exact Masking Contract

The masking operation should be treated as part of the model interface, not as an informal augmentation. Let the patch sequence be indexed by:

$$
\mathcal{I}=\{1,\ldots,N\}.
$$

A random permutation $\pi$ determines the visible prefix and masked suffix:

$$
\mathcal{V}=\{\pi_1,\ldots,\pi_{(1-r)N}\},
\qquad
\mathcal{M}=\{\pi_{(1-r)N+1},\ldots,\pi_N\}.
$$

The encoder receives an ordered sequence of visible patches, while the decoder restores the original spatial positions. A faithful implementation therefore has to preserve both pieces of information:

1. which patches survived the random mask;
2. where every visible and masked patch belongs in the original grid.

An implementation that drops the restore indices or applies positional embeddings after an incorrect reorder is no longer implementing the same architecture. The usual data path is:

$$
X
\xrightarrow{\text{patchify}}
E
\xrightarrow{\text{shuffle and gather}}
E_{\mathcal{V}}
\xrightarrow{\text{encoder}}
Z_{\mathcal{V}}
\xrightarrow{\text{append mask tokens and unshuffle}}
Z_{\text{full}}.
$$

The mask ratio changes both the learning problem and the compute budget. It should therefore be logged as a first-class experiment parameter, alongside patch size, image resolution, and decoder width.

## Decoder Assembly

Let $m$ be a learned decoder mask token. The encoder output is projected to the decoder width when the two widths differ:

$$
\tilde{Z}_{\mathcal{V}}=Z_{\mathcal{V}}W_{\text{enc}\rightarrow\text{dec}}.
$$

For each masked position, insert a copy of $m$ and then restore the original patch order:

$$
U_i=
\begin{cases}
\tilde{z}_i, & i\in\mathcal{V},\\
m, & i\in\mathcal{M}.
\end{cases}
$$

The decoder input is:

$$
Z_{\text{dec}}=U+P_{\text{dec}}.
$$

The position embedding is important because a masked token by itself carries no location. The decoder must infer content at a known spatial coordinate, not merely produce an unordered set of plausible patches.

The decoder is intentionally smaller than the encoder. This gives the reconstruction target enough capacity to provide a learning signal without making the pretraining helper the dominant computation:

$$
\text{large encoder}\;\gg\;\text{light decoder}.
$$

This asymmetry also explains why the decoder is discarded after pretraining. It is optimized for exposing missing-patch information, while the encoder is optimized for a reusable representation.

## Target Normalization

The reconstruction target can be formed from raw patch pixels or normalized patch pixels. With per-patch normalization:

$$
\bar{x}_i=\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},
$$

the loss becomes:

$$
\mathcal{L}_{\text{masked}}
=
\frac{1}{|\mathcal{M}|}
\sum_{i\in\mathcal{M}}
\left\lVert
\hat{x}_i-\bar{x}_i
\right\rVert_2^2.
$$

This choice changes what the decoder is asked to model. Raw pixels preserve absolute brightness and color statistics; normalized targets emphasize the patch's local structure. When reproducing a result, target normalization cannot be treated as a cosmetic preprocessing detail.

## Pretraining to Downstream Interface

MAE has two distinct model graphs:

| Stage | Graph | Output used |
| --- | --- | --- |
| pretraining | visible-only encoder plus full-sequence lightweight decoder | masked-patch reconstruction loss |
| transfer | encoder on the complete patch sequence | class token or pooled representation |

During transfer, the encoder normally sees all patches because there is no reconstruction mask. A classification head is attached for fine-tuning, or the encoder output is frozen for a linear probe. Comparing these settings matters:

$$
\text{linear probe}\neq\text{partial fine-tuning}\neq\text{full fine-tuning}.
$$

The pretraining decoder should not accidentally remain in the downstream graph. Keeping it can inflate memory and obscure whether the encoder itself learned a useful representation.

## Why the Architecture Scales

The efficiency argument has two parts. First, the encoder's quadratic attention sees fewer tokens:

$$
\frac{C_{\text{enc,MAE}}}{C_{\text{enc,dense}}}
\approx
(1-r)^2
$$

for the attention-dominated component. Second, the encoder does not spend layers transforming mask tokens that carry no observed content. The decoder pays a smaller full-sequence cost once, with fewer layers and a narrower hidden size.

This allocation is especially useful as the backbone grows. If the decoder were as large as the encoder, masked reconstruction would lose much of its computational advantage. If the decoder were too weak, the target would provide a poor gradient signal. Decoder width and depth are therefore meaningful scaling variables, not arbitrary implementation choices.

## Ablations Worth Reproducing

| Ablation | Question | Expected interpretation |
| --- | --- | --- |
| mask ratio | Is the task too easy or too underdetermined? | low ratios permit local copying; very high ratios may remove too much context |
| decoder capacity | Is the reconstruction head bottlenecking learning? | a small decoder should work, but an excessively weak one can underfit targets |
| pixel normalization | What information should the target emphasize? | normalized targets alter the balance between local texture and global structure |
| encoder visibility | Does the encoder receive mask tokens? | visible-only input is the central compute-saving design |
| patch size | How coarse is the prediction unit? | smaller patches increase sequence length and detail; larger patches reduce tokens |
| fine-tuning protocol | Is the representation or recipe responsible for the gain? | probe and fine-tune results answer different questions |

The most informative baseline is not only a different loss. It is a dense ViT with a comparable optimizer, augmentation policy, training budget, and downstream recipe. Otherwise the comparison confounds architecture with compute and tuning.

## Failure Modes

1. **Mask leakage**: masked patch values reach the encoder through an incorrect gather, residual input, or augmentation cache.
2. **Position mismatch**: visible tokens are restored in the wrong order, so the decoder receives content at the wrong coordinates.
3. **Visible-patch loss**: the loss includes unmasked patches and rewards copying rather than inference.
4. **Decoder dominance**: a decoder close in size to the encoder erases the intended compute asymmetry.
5. **Target mismatch**: normalized targets are used in one run and raw pixels in another without being recorded.
6. **Transfer confusion**: a full fine-tuning result is described as evidence from a frozen representation.

For debugging, first visualize the mask and restored patch order, then overfit a tiny batch. A correct implementation should reduce masked reconstruction loss on the tiny batch before large-scale training is attempted.

## Comparison with DINO

MAE and DINO use related ViT backbones but impose different learning signals:

| Axis | MAE | DINO |
| --- | --- | --- |
| observed input | visible subset | multiple augmented views |
| target | masked pixels | teacher representation distribution |
| decoder | lightweight reconstruction decoder | projection heads for teacher and student |
| central difficulty | infer missing spatial content | align views without collapse |
| encoder efficiency | fewer tokens during pretraining | repeated crops and teacher-student passes |
| transferred signal | reconstruction-trained encoder | self-distilled semantic representation |

The comparison prevents a common category error: masked reconstruction and self-distillation are both self-supervised, but they shape representations through different target spaces. See [[papers/architectures/emerging-properties-in-self-supervised-vision-transformers|DINO]] and [[concepts/learning/self-supervised-learning|Self-supervised learning]].

## Transfer Beyond Natural Images

The MAE pattern can be transferred when an input has meaningful local units and a valid partial-observation task:

$$
\text{structured object}
\rightarrow
\text{mask units}
\rightarrow
\text{encode observed units}
\rightarrow
\text{reconstruct hidden units}.
$$

For molecular or structural data, the unit might be a residue neighborhood, atom neighborhood, spatial crop, or modality-specific token. The transfer is not automatic. A random mask must preserve a meaningful conditional prediction problem, and the reconstruction target should not reward trivial coordinate or padding shortcuts. Useful checks include:

- whether masked units are spatially connected or randomly scattered;
- whether the target has symmetries that require invariant or equivariant decoding;
- whether the mask can be reconstructed from leakage in ordering or metadata;
- whether the downstream split is family-, scaffold-, or time-separated;
- whether reconstruction quality correlates with the intended downstream property.

The reusable idea is the asymmetric information bottleneck, not the literal image-pixel target.

## Reproduction Checklist

- [ ] confirm the patchification convention and number of patches;
- [ ] record mask ratio, random seed, and mask sampling implementation;
- [ ] verify that the encoder receives visible patches only;
- [ ] verify that decoder tokens are restored to original positions;
- [ ] record decoder depth, width, and projection layer;
- [ ] record raw versus normalized reconstruction targets;
- [ ] verify that the loss is masked-only;
- [ ] separate pretraining, linear-probe, and fine-tuning code paths;
- [ ] compare compute and data budgets with the baseline;
- [ ] run a tiny-batch overfit test before a full run.

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
