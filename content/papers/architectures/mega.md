---
title: Mega
aliases:
  - papers/mega
  - papers/moving-average-equipped-gated-attention
tags:
  - papers
  - architectures
  - attention
  - gating
  - sequence-modeling
  - long-context
---

# Mega

> Mega combines gated attention with exponential moving average so that attention gains a local sequential inductive bias while retaining flexible content-dependent sequence mixing.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Mega: Moving Average Equipped Gated Attention |
| Authors | Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, Luke Zettlemoyer |
| Year | 2022 preprint; 2023 conference |
| Venue | ICLR 2023 |
| arXiv | [2209.10655](https://arxiv.org/abs/2209.10655) |
| OpenReview | [qNLe3iq2El](https://openreview.net/forum?id=qNLe3iq2El) |
| Code | [facebookresearch/mega](https://github.com/facebookresearch/mega) |
| Status | full note started |

## Question

Self-attention is flexible because it learns pairwise interactions:

$$
A(X)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right),
\qquad
Y=A(X)V.
$$

But this flexibility comes with two pressures:

| Pressure | Meaning |
| --- | --- |
| weak local inductive bias | attention does not assume nearby tokens should matter more |
| quadratic sequence cost | full attention materializes or computes an $L\times L$ interaction pattern |

Mega asks:

$$
\text{Can attention keep flexible content mixing while gaining a local sequential bias?}
$$

Its answer is:

$$
\text{exponential moving average}
+
\text{single-head gated attention}.
$$

## Main Claim

Mega's durable architecture claim is:

$$
\text{attention}
+
\text{EMA local bias}
+
\text{gating}
\Rightarrow
\text{strong long-sequence sequence mixer}.
$$

The paper positions Mega against Transformers and recent state-space models. It reports results across Long Range Arena, machine translation, autoregressive language modeling, image classification, and raw speech classification.

The important architecture point is not only the benchmark table. It is that Mega makes local memory an explicit part of the attention block:

$$
X
\xrightarrow{\operatorname{EMA}}
\tilde{X}
\xrightarrow{\operatorname{GatedAttention}}
Y.
$$

This makes Mega a bridge between [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], [[papers/architectures/s4|S4]], [[papers/architectures/hyena|Hyena]], [[papers/architectures/retnet|RetNet]], and [[papers/architectures/mamba|Mamba]].

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | ordered token or feature sequence |
| Output | contextual sequence representation |
| Target replacement | Transformer self-attention block |
| Local bias | exponential moving average over sequence positions |
| Content mixing | gated single-head attention |
| Efficiency variant | Mega-chunk with fixed-size chunks |
| Main comparison | Transformer variants, S4, long-sequence sequence models |

Mega is not a pure RNN, not a pure SSM, and not a standard Transformer. It keeps attention-style content mixing but adds a recurrent/convolution-like local smoothing path.

## Exponential Moving Average

The basic EMA recurrence is:

$$
z_t
=
\alpha x_t
+
(1-\alpha)z_{t-1}.
$$

Equivalently, the current output is a decayed weighted sum of past inputs:

$$
z_t
=
\sum_{i=1}^{t}
\alpha(1-\alpha)^{t-i}x_i.
$$

This has an explicit positional bias:

$$
\text{older token influence}
\propto
(1-\alpha)^{\Delta t}.
$$

The bias is useful for local sequential structure, but a fixed EMA alone is too rigid for general sequence modeling. Mega therefore uses a more flexible multi-dimensional damped EMA and feeds the result into gated attention.

## Why EMA Helps Attention

Attention learns dependencies from data:

$$
A_{t,i}
=
\operatorname{softmax}_i(q_t^\top k_i).
$$

EMA provides a decayed local summary:

$$
z_t = \operatorname{EMA}(x_{1:t}).
$$

Mega uses the EMA path to create position-aware representations before the attention-style operation. In plain terms:

| Mechanism | What It Adds |
| --- | --- |
| attention | content-dependent interaction |
| EMA | local sequential inductive bias |
| gating | multiplicative control over information flow |
| chunking | lower sequence complexity for long inputs |

The combination is different from simply adding relative position bias. EMA gives a stateful local smoothing operator, not only a score offset.

## Gated Attention View

Mega uses a gated attention unit rather than standard multi-head attention. A simplified block can be read as:

$$
\tilde{X} = \operatorname{EMA}(X)
$$

$$
U,V,G = \operatorname{Proj}(\tilde{X})
$$

$$
A = \operatorname{Attention}(U,V)
$$

$$
Y = G\odot A.
$$

The exact paper formulation has more detail, but the reusable architecture idea is:

$$
\text{local memory}
\rightarrow
\text{single-head gated attention}
\rightarrow
\text{sequence representation}.
$$

This is why Mega should be read together with [[concepts/architectures/gating|Gating]], [[concepts/architectures/attention|Attention]], and [[concepts/architectures/convolution|Convolution]].

## Mega-Chunk

Full attention over length $L$ has quadratic cost:

$$
O(L^2).
$$

Mega-chunk reduces the cost by splitting the sequence into fixed-size chunks:

$$
X=[C_1,C_2,\ldots,C_m].
$$

Within chunks, the model applies gated attention. Across the sequence, the EMA path helps preserve contextual continuity:

$$
\text{chunk attention}
+
\text{moving-average state}.
$$

This gives a useful architecture pattern:

$$
\text{local flexible interaction}
+
\text{global decayed memory}.
$$

The paper reports that this linear-time variant loses little quality relative to full Mega under its evaluated settings.

## Relation to Transformer, S4, Hyena, RetNet, and Mamba

| Paper | Core Mixer | Main Bias |
| --- | --- | --- |
| [Transformer](/papers/architectures/attention-is-all-you-need) | multi-head self-attention | flexible pairwise token interaction |
| [S4](/papers/architectures/s4) | structured state-space layer | long-range linear state dynamics |
| [Hyena](/papers/architectures/hyena) | gated implicit long convolution | long convolution plus data-controlled gates |
| [Mega](/papers/architectures/mega) | EMA-equipped gated attention | local decayed memory plus content mixing |
| [RetNet](/papers/architectures/retnet) | retention | decayed key-value state with recurrent form |
| [Mamba](/papers/architectures/mamba) | selective SSM scan | input-dependent state-space recurrence |

Mega is especially useful for understanding a design pattern that later appears in several forms:

$$
\text{sequence state or decay}
+
\text{content-dependent gate}
+
\text{attention-like or scan-like mixing}.
$$

It is not the same as Mamba or RetNet, but it belongs in the same long-sequence architecture shelf.

## Evidence to Read Carefully

| Evidence | What It Supports | What To Check |
| --- | --- | --- |
| Long Range Arena results | long-sequence modeling quality | task-specific inductive bias and baseline tuning |
| machine translation and language modeling | general sequence modeling beyond synthetic tasks | training budget and architecture parity |
| image and speech classification | modality transfer of sequence block | tokenization and preprocessing differences |
| Mega-chunk results | chunking can reduce cost with limited quality loss | chunk size, context boundary, and memory path |
| comparison to S4 and Transformer variants | broad competitiveness | whether implementation maturity differs |

The key audit is:

$$
\text{Mega improvement}
=
\text{EMA bias}
+
\text{gating}
+
\text{attention design}
+
\text{training setup}.
$$

Do not attribute every result to a single component unless an ablation isolates it.

## Practical Interpretation

Use Mega as a reference when reading papers that:

- add moving-average, decay, or local memory to attention;
- replace multi-head attention with gated single-head variants;
- combine chunked attention with recurrent or convolution-like memory;
- compare attention against SSMs or long-convolution sequence models;
- claim long-sequence improvements from local inductive bias.

Mega is a useful reminder that not all efficient sequence models remove attention. Some keep attention but make its inductive bias less blank.

## Limitations

- EMA favors local decayed dependencies, which may not match every task.
- Chunking changes information flow at chunk boundaries.
- Single-head gated attention should be compared against well-tuned multi-head baselines.
- Results across modalities depend on tokenization, resolution, sequence length, and training recipe.
- Later SSM/selective-scan models may have different hardware behavior and scaling profiles.

## Why It Belongs in Architecture Papers

Mega contributes a reusable sequence block:

$$
\operatorname{MegaBlock}
=
\operatorname{GatedAttention}
\circ
\operatorname{EMA}.
$$

The paper is not merely an evaluation or application paper. It adds a concrete architectural primitive for long-sequence modeling.

## Related

- [[ai/architectures|Architectures]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/hyena|Hyena]]
- [[papers/architectures/retnet|RetNet]]
- [[papers/architectures/mamba|Mamba]]
