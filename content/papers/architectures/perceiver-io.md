---
title: Perceiver IO
aliases:
  - papers/perceiver-io
tags:
  - papers
  - architectures
  - attention
  - multimodal
---

# Perceiver IO

> The paper generalizes attention models by routing large structured inputs and outputs through a fixed-size latent array.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Perceiver IO: A General Architecture for Structured Inputs & Outputs |
| Authors | Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Henaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, Joao Carreira |
| Year | 2021 |
| Venue | ICLR 2022 |
| arXiv | [2107.14795](https://arxiv.org/abs/2107.14795) |
| OpenReview | [fILj7WpI-g](https://openreview.net/forum?id=fILj7WpI-g) |
| Status | full note started |

## One-Line Takeaway

[[papers/architectures/attention-is-all-you-need|Transformers]] scale poorly when attention is applied directly to huge input/output arrays; Perceiver IO keeps the expensive recurrent computation in a fixed-size latent array and uses cross-attention to interface with arbitrary inputs and outputs.

## Question

Many domains can be written as arrays:

- text tokens;
- image patches or pixels;
- audio samples;
- video frames;
- point sets;
- multimodal streams;
- dense output grids;
- class logits;
- optical-flow fields;
- game states and action distributions.

The standard Transformer is flexible, but self-attention over $n$ input tokens costs:

$$
O(n^2d)
$$

for hidden width $d$. This becomes expensive when $n$ is very large, as in high-resolution images, video, long audio, or dense multimodal observations.

Domain-specific models solve this by using inductive biases:

- CNNs use locality and translation sharing;
- RNNs use recurrence over sequence steps;
- GNNs use graph neighborhoods;
- U-Nets use hierarchical image grids;
- point-cloud models use permutation-aware set operations.

The paper asks:

> Can one architecture process large structured inputs and produce structured outputs without building a separate backbone for every modality?

## Main Claim

Perceiver IO separates three jobs:

1. encode raw inputs into an input array;
2. move information from the input array into a fixed-size latent array;
3. decode task-specific outputs by querying the latent array.

The high-level computation is:

$$
X
\xrightarrow{\text{input adapter}}
\tilde{X}
\xrightarrow{\text{input cross-attention}}
Z
\xrightarrow{\text{latent processing}}
Z_L
\xrightarrow{\text{output query cross-attention}}
Y.
$$

where:

- $X$ is raw task input;
- $\tilde{X} \in \mathbb{R}^{n \times d_x}$ is the input token/array representation;
- $Z \in \mathbb{R}^{m \times d_z}$ is a learned latent array;
- $m$ is fixed or at least much smaller than $n$;
- $Y$ is an output array specified by output queries.

The core idea:

$$
n \text{ can be large, but } m \text{ is controlled.}
$$

Most repeated self-attention happens over the latent size $m$, not the input size $n$.

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Input adapter | raw modality data | input array $\tilde{X}$ | domain formatting |
| Input cross-attention | latent queries $Z$, input keys/values $\tilde{X}$ | updated latents | ingest information |
| Latent Transformer | latent array | processed latents | main computation |
| Output queries | desired output positions/tasks | query array $Q_Y$ | define output structure |
| Output cross-attention | output queries, latent keys/values | output array | decode structured predictions |

This makes the architecture modular:

$$
\text{modality-specific adapters}
+
\text{modality-agnostic latent core}
+
\text{task-specific queries}.
$$

## Input Cross-Attention

Let the input array be:

$$
X \in \mathbb{R}^{n \times d_x},
$$

and the latent array be:

$$
Z \in \mathbb{R}^{m \times d_z}.
$$

The input cross-attention uses latents as queries and inputs as keys/values:

$$
Q = ZW^Q,
$$

$$
K = XW^K,
$$

$$
V = XW^V.
$$

Then:

$$
\operatorname{CrossAttn}(Z,X)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V.
$$

The attention matrix has shape:

$$
m \times n.
$$

So input ingestion costs roughly:

$$
O(mnd).
$$

This is the key scaling move. Direct self-attention on $X$ would cost:

$$
O(n^2d).
$$

When $m \ll n$:

$$
O(mnd) \ll O(n^2d).
$$

## Latent Self-Attention

After the input is read into the latent array, Perceiver IO applies repeated Transformer blocks to $Z$:

$$
Z_{\ell+1}
=
\operatorname{TransformerBlock}(Z_\ell).
$$

Each latent self-attention layer costs:

$$
O(m^2d).
$$

If the model uses $L$ latent layers:

$$
O(Lm^2d).
$$

Because $m$ is fixed by architecture design, not by raw input length, the expensive depth of computation is bounded.

The full rough cost is:

$$
O(mnd) + O(Lm^2d) + O(qmd),
$$

where $q$ is the number of output queries.

## Output Query Decoding

The original Perceiver already used a latent bottleneck for input processing. Perceiver IO adds a more general output interface.

Let output queries be:

$$
Q_Y \in \mathbb{R}^{q \times d_q}.
$$

These queries specify what the model should output. They can represent:

- class queries;
- pixel coordinates;
- audio time positions;
- language positions;
- optical-flow locations;
- arbitrary task slots.

Output decoding cross-attends from output queries to processed latents:

$$
Q = Q_Y W^Q,
$$

$$
K = Z_L W^K,
$$

$$
V = Z_L W^V.
$$

Then:

$$
Y
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V.
$$

The output attention matrix has shape:

$$
q \times m.
$$

So output decoding costs:

$$
O(qmd).
$$

This is what makes Perceiver IO different from a plain latent encoder. The output array is not restricted to a single pooled vector.

## Input and Output Arrays

Perceiver IO treats both input and output as arrays. This is a broad abstraction.

| Task Type | Input Array | Output Query |
| --- | --- | --- |
| Classification | image/audio/text tokens | class query or label slots |
| Dense prediction | pixels or patches | output pixel/location queries |
| Language modeling | token array | target-position queries |
| Multimodal learning | concatenated modality arrays | task-specific queries |
| Optical flow | paired image features | spatial flow queries |
| Game state modeling | observation tokens | action/value queries |

The architecture does not remove task-specific formatting. It moves that formatting to adapters and queries.

## Positional and Modality Encoding

Attention alone is content-based. For structured arrays, the model usually needs extra information:

$$
\tilde{x}_i = \operatorname{Embed}(x_i) + p_i + t_i,
$$

where:

- $p_i$ is a position, coordinate, or Fourier feature;
- $t_i$ can mark modality or source type;
- $\operatorname{Embed}(x_i)$ maps raw values to hidden features.

For images:

$$
p_i = (u_i, v_i)
$$

or a Fourier encoding of the pixel/patch coordinate.

For audio:

$$
p_i = t_i
$$

or a time/frequency coordinate.

For multimodal arrays, modality embeddings can indicate whether an element comes from text, image, audio, or another source.

## Why the Latent Array Matters

The latent array is a learned computational workspace:

$$
Z_0 \in \mathbb{R}^{m \times d}.
$$

It is not the same as:

- a sequence of input tokens;
- a class token;
- a pooled vector;
- a set of fixed labels.

The latent slots repeatedly exchange information through self-attention:

$$
Z_{\ell+1}
=
\operatorname{SelfAttn}(Z_\ell)
+
\operatorname{FFN}(Z_\ell)
$$

up to residual and normalization details.

This gives the model a controlled internal memory size. The price is information bottleneck risk:

$$
X \in \mathbb{R}^{n \times d_x}
\rightarrow
Z \in \mathbb{R}^{m \times d_z}.
$$

If $m$ is too small or the queries do not read the right information, the bottleneck can discard details.

## Comparison to Standard Transformer

| Property | Standard Transformer Encoder | Perceiver IO |
| --- | --- | --- |
| Main self-attention domain | input tokens | latent array |
| Input scaling | $O(n^2d)$ | $O(mnd) + O(m^2d)$ |
| Output interface | often pooled token or sequence positions | arbitrary output queries |
| Modality handling | tokenization plus positional encoding | input adapters plus positional/modality encoding |
| Bottleneck | none unless added | explicit latent array |
| Strength | simple and direct | scales to large structured arrays |
| Risk | expensive for large $n$ | latent bottleneck may lose information |

For normal-length text, a standard Transformer can be simpler. For high-dimensional dense data, Perceiver IO's latent bottleneck can be attractive.

## Comparison to Set Transformer

| Property | [[papers/architectures/set-transformer|Set Transformer]] | Perceiver IO |
| --- | --- | --- |
| Main input type | unordered sets | structured arrays across modalities |
| Bottleneck mechanism | inducing points and PMA seeds | latent array and output queries |
| Symmetry emphasis | permutation invariance/equivariance | flexible input/output formatting |
| Output interface | PMA slots | arbitrary output queries |
| Scaling move | ISAB reduces set attention to $O(nm)$ | input cross-attention plus latent self-attention |
| Typical use | set-level prediction, clustering, set functions | multimodal and dense structured prediction |

The family resemblance is strong: both use learned query arrays to avoid direct full attention over all raw elements. The emphasis differs. Set Transformer is primarily about set symmetry; Perceiver IO is primarily about general structured input/output scaling.

## Comparison to Encoder-Decoder Attention

Perceiver IO can be read as a general encoder-decoder architecture:

$$
\text{input array}
\rightarrow
\text{latent memory}
\rightarrow
\text{output queries}.
$$

This resembles cross-attention in sequence-to-sequence models, but with a fixed latent core:

| Part | Seq2seq Transformer | Perceiver IO |
| --- | --- | --- |
| Encoder memory | input token states | latent array |
| Decoder query | target prefix tokens | output queries |
| Main scaling issue | source and target length | input/output array size |
| Task structure | usually sequence-to-sequence | arbitrary arrays |

The output queries can be learned, coordinate-derived, or task-provided.

## Evidence Reading

The paper demonstrates the same architecture family across multiple task types. The evidence supports architectural generality more than domain-specific superiority.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Perceiver IO can handle varied modalities | experiments across vision, language, multimodal, and structured outputs | one architecture template can be reused | adapters still differ |
| Latent bottleneck enables scaling | comparisons to direct attention-style processing | large inputs can be processed with bounded latent compute | quality depends on latent size |
| Output queries support structured outputs | dense or task-specific output experiments | outputs need not be a single class vector | query construction matters |
| Architecture can compete with specialized models | benchmark results | generalist design is viable | best specialized models may still win |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | structured input/output modeling |
| Input unit | arbitrary array elements |
| Output unit | arbitrary query-defined array |
| Main mechanism | cross-attention into and out of a latent array |
| Main scaling claim | repeated computation happens in fixed latent space |
| Main comparison | Transformers and domain-specific models |
| Key hyperparameters | latent size $m$, latent width $d$, latent depth $L$, output query count $q$ |
| Not the claim | no-free-lunch replacement for all domain-specific backbones |

## Architecture as a Design Pattern

Perceiver IO is more than one model. It is a design pattern:

$$
\text{Large input}
\xrightarrow{\text{cross-attend}}
\text{small latent workspace}
\xrightarrow{\text{compute}}
\text{query-defined output}.
$$

This pattern appears in later systems as:

- latent bottleneck multimodal encoders;
- query-based detection heads;
- memory tokens;
- resampler modules;
- learned pooling modules;
- slot-based decoders.

The recurring question is:

$$
\text{What information must fit through the bottleneck?}
$$

## How to Choose Latent Size

The latent size $m$ controls compute and capacity.

If $m$ is too small:

$$
Z \text{ cannot store enough task-relevant information.}
$$

If $m$ is too large:

$$
O(mnd) + O(Lm^2d)
$$

can become expensive, and the model loses part of its scaling advantage.

Practical reading questions:

- Does performance saturate as $m$ increases?
- Is the task dense enough to need many latents?
- Are outputs localized or global?
- Does the adapter compress too much before cross-attention?
- Is the latent array reused across input blocks or refreshed?

## Molecular and Structural Modeling Reading

Perceiver IO is not a molecular architecture by itself, but its pattern is useful for molecular and structural AI.

Possible uses:

- combine protein sequence tokens, structure tokens, ligand atom features, and pocket descriptors;
- process large multiple sequence alignment-like arrays through a latent bottleneck;
- decode per-residue, per-atom, pairwise, or global outputs using output queries;
- aggregate a large set of docking poses, conformers, or candidate complexes;
- build a task interface where different prediction heads query the same latent molecular context.

Potential mismatch:

- plain attention does not guarantee $SE(3)$ equivariance;
- atom/bond topology may require graph-aware encoders;
- pairwise geometric outputs may need explicit pair representations;
- dense coordinate prediction can be sensitive to output query design;
- chemical validity is not enforced by the latent bottleneck.

A reasonable hybrid pattern is:

$$
\text{domain encoder}
\rightarrow
\text{array of structure/molecule features}
\rightarrow
\text{Perceiver-style latent bottleneck}
\rightarrow
\text{task-specific output queries}.
$$

## Implementation Notes

### Masking

Variable-size arrays need masks. If padded input positions are not masked, cross-attention can leak padding artifacts.

For input cross-attention:

$$
A_{ij}
=
\begin{cases}
\frac{q_i^\top k_j}{\sqrt{d_h}}, & \text{valid input } j, \\
-\infty, & \text{padded input } j.
\end{cases}
$$

Then:

$$
\alpha_{ij} = \operatorname{softmax}_j(A_{ij}).
$$

### Output Query Semantics

Output queries are part of the model specification. For a dense grid output, a query might include coordinate features:

$$
q_{u,v} = \operatorname{Embed}(u,v).
$$

For class outputs, it may be learned:

$$
q_c \in \mathbb{R}^{d}.
$$

For multi-task outputs, queries may include task IDs:

$$
q_{t,i} = \operatorname{Embed}(\text{task}=t, \text{position}=i).
$$

Changing output queries changes the model contract.

### Adapter Responsibility

The input adapter is not a trivial detail. It decides what raw information is visible to the latent array.

For an image, the adapter may expose:

- pixels;
- patches;
- convolutional features;
- Fourier position encodings.

For a molecule, the adapter may expose:

- atom type features;
- bond features;
- residue features;
- pair distances;
- coordinate encodings;
- graph or geometric encoder outputs.

If the adapter hides important structure, the latent core cannot recover it.

### Bottleneck Diagnostics

Signs that the latent bottleneck is too tight:

- performance improves strongly with $m$;
- outputs miss fine spatial details;
- attention maps concentrate on a few broad regions;
- dense prediction quality is worse than global classification quality;
- the model underfits even with sufficient depth.

## Common Misreadings

### "Perceiver IO removes modality-specific engineering."

It reduces the need for separate backbones, but it still needs input adapters, positional encodings, modality markers, output queries, and losses.

### "The latent array is just a class token."

A class token is usually one token inside the input sequence. Perceiver latents are a separate learned array that repeatedly self-attends and acts as a computational workspace.

### "It is always cheaper than a Transformer."

Only when $m$ is sufficiently smaller than $n$ and the cross-attention/adapter overhead does not dominate. For small inputs, a standard Transformer may be simpler and faster.

### "General architecture means best architecture."

The paper argues for broad applicability. It does not prove that one general architecture dominates specialized architectures in every domain.

## Later-Paper Checklist

When reading a later multimodal, memory-token, resampler, query-decoder, or latent-bottleneck paper, ask:

- What is the input array?
- What is the latent array size?
- Which computations happen over input length $n$ and which over latent length $m$?
- Are output queries learned, coordinate-derived, task-provided, or autoregressive?
- Does the bottleneck lose information needed for dense outputs?
- Are modality and position encodings doing most of the domain work?
- Is the comparison against direct attention compute-matched?
- Is the architecture generality claim tested across genuinely different task structures?

## Why It Matters

Perceiver IO is important because it reframes attention as an interface between arrays, not only as a sequence model block. It makes three ideas explicit:

- large inputs can be read through cross-attention;
- repeated computation can happen in a smaller latent workspace;
- outputs can be specified by queries.

Those ideas show up repeatedly in modern multimodal, detection, segmentation, retrieval, and agent-memory architectures.

## Limitations

- The latent bottleneck can discard details.
- Output query design is a task-specific modeling choice.
- Input adapters still carry domain assumptions.
- Dense output tasks can be sensitive to positional encoding and query resolution.
- Specialized architectures may be more efficient or accurate under matched budgets.
- The architecture can be harder to debug than a plain encoder or decoder Transformer.

## Connections

- [[concepts/architectures/perceiver|Perceiver]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/modalities/index|Modalities]]
- [[papers/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/deep-sets|Deep Sets]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
