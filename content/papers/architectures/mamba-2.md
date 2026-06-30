---
title: Mamba-2
aliases:
  - papers/mamba-2
  - papers/transformers-are-ssms
  - papers/structured-state-space-duality
tags:
  - papers
  - architectures
  - state-space-model
  - mamba
  - attention
  - sequence-modeling
---

# Mamba-2

> The paper introduces structured state space duality, connecting SSMs and attention through structured semiseparable matrices, and uses that view to design the faster Mamba-2 architecture.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality |
| Authors | Tri Dao, Albert Gu |
| Year | 2024 |
| Venue | ICML 2024 |
| arXiv | [2405.21060](https://arxiv.org/abs/2405.21060) |
| Code | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| Status | full note started |

## Question

Mamba showed that a selective state-space model can be a serious alternative to attention for sequence modeling:

$$
h_t
=
A_t h_{t-1}
+
B_t x_t,
\qquad
y_t=C_t h_t.
$$

Transformers use a very different surface form:

$$
Y
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V.
$$

Mamba-2 asks a deeper architecture question:

$$
\text{Are SSMs and attention separate families, or two views of a shared structured matrix mixer?}
$$

The paper's answer is structured state space duality:

$$
\text{structured SSM recurrence}
\quad
\leftrightarrow
\quad
\text{structured masked attention-like matrix}.
$$

## Main Claim

The durable claim is not merely that Mamba-2 is faster. The more important claim is:

$$
\text{SSMs and attention can be related through structured semiseparable matrices}.
$$

This framework leads to a new core layer:

$$
\text{SSD layer}
\rightarrow
\text{Mamba-2}.
$$

The paper reports that Mamba-2 refines Mamba's selective SSM core layer and is 2-8x faster while remaining competitive with Transformers on language modeling.

For this wiki, the key architecture lesson is:

$$
\text{architecture families should be compared by their sequence mixing matrix, not only by their implementation form}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Output | contextual token representations or language-model logits |
| Main object | sequence mixing operator |
| Theoretical bridge | structured state space duality |
| Matrix class | structured semiseparable matrices |
| New architecture | Mamba-2 / SSD layer |
| Practical target | faster SSM-style sequence modeling while staying competitive with Transformer language models |

Mamba-2 belongs near [[papers/architectures/mamba|Mamba]], [[papers/architectures/s4|S4]], [[papers/architectures/retnet|RetNet]], [[papers/architectures/mega|Mega]], [[papers/architectures/hyena|Hyena]], and [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]].

## Matrix Mixer View

A causal sequence layer can often be read as:

$$
Y = MX,
$$

where $M$ is a lower-triangular mixing matrix. Different architectures define $M$ differently.

| Family | Mixer View |
| --- | --- |
| attention | content-dependent token-token matrix |
| linear attention | kernelized low-rank or recurrent form |
| SSM | recurrence-induced structured lower-triangular matrix |
| retention | decayed key-value matrix with recurrent form |
| Mamba-2 | SSD matrix mixer with efficient dual algorithms |

For an SSM, the recurrence expands into:

$$
y_t
=
\sum_{i\le t}
C_t
\left(
\prod_{j=i+1}^{t}A_j
\right)
B_i x_i.
$$

This means the sequence layer has implicit entries:

$$
M_{t,i}
=
C_t
\left(
\prod_{j=i+1}^{t}A_j
\right)
B_i,
\qquad
i\le t.
$$

That is the bridge: recurrence defines a structured causal matrix.

## Structured Semiseparable Matrices

The paper connects SSMs and attention-like mechanisms through structured semiseparable matrices.

The useful reading is:

$$
M_{t,i}
\text{ is not arbitrary;}
\quad
M
\text{ has low-rank structure in its off-diagonal blocks.}
$$

This matters because efficient algorithms can exploit the structure. The same sequence transformation may have:

| View | Strength |
| --- | --- |
| recurrent view | streaming and linear-time inference |
| matrix view | parallel training and relationship to attention |
| block/chunk view | hardware-aware efficient computation |

This is the same kind of lesson as [[papers/architectures/flashattention|FlashAttention]]: a model family is partly mathematical and partly algorithmic.

## Mamba-2 vs Mamba

Mamba-1 introduced selective SSMs:

$$
h_t
=
A_t h_{t-1}
+
B_t x_t,
\qquad
y_t=C_t h_t.
$$

Mamba-2 refines the core layer using SSD. The paper's practical claim is that this refinement improves algorithmic efficiency while preserving the sequence modeling role of Mamba.

| Axis | Mamba | Mamba-2 |
| --- | --- | --- |
| core story | selective SSM | structured state space duality |
| comparison target | attention alternatives | common framework for SSMs and attention variants |
| implementation goal | hardware-aware selective scan | faster SSD/Mamba-2 layer |
| architecture lesson | input-dependent state update matters | sequence mixers can have dual recurrent and matrix forms |

This makes Mamba-2 more than a parameter tweak. It changes how to reason about the family.

## Relation to Attention

Attention computes:

$$
Y
=
A(X)V,
\qquad
A(X)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right).
$$

Mamba-2 does not say standard softmax attention and every SSM are identical. The useful claim is narrower:

$$
\text{certain structured attention-like mixers}
\leftrightarrow
\text{certain structured SSM mixers}.
$$

That lets the paper transfer intuition and algorithms between:

- linear attention;
- gated or decayed attention-like mechanisms;
- selective SSMs;
- block/chunk recurrent computation.

For reading architecture papers, this is valuable because it prevents a false binary:

$$
\text{attention}
\quad\text{vs}\quad
\text{state}.
$$

Often the real distinction is:

$$
\text{what structure does the sequence mixing matrix have?}
$$

## Algorithmic Lesson

Mamba-2 is an architecture paper, but its core contribution is inseparable from algorithms.

The same abstract operator can be computed through different paths:

| Computation Path | When It Matters |
| --- | --- |
| recurrence | streaming inference |
| parallel scan | training throughput |
| chunk/block algorithm | long sequences and accelerator memory hierarchy |
| matrix form | theory, comparison to attention, and algorithm design |

This is why Mamba-2 belongs next to [[papers/architectures/flashattention|FlashAttention]] in an architecture wiki: both show that the usable architecture is shaped by hardware-aware algorithms.

## Evidence to Read Carefully

| Evidence | What It Supports | What To Check |
| --- | --- | --- |
| SSD theory | SSMs and attention variants share structured matrix views | the assumptions on matrix structure |
| Mamba-2 layer benchmarks | faster core layer than Mamba under reported settings | hardware, sequence length, batch, implementation maturity |
| language modeling results | Mamba-2 remains competitive with Transformer baselines | training data, token budget, model size, optimizer recipe |
| algorithmic speedups | SSD computation can improve practical throughput | whether end-to-end training/inference includes all overhead |

The paper's broad title should be read carefully. It is not a license to collapse all Transformers and all SSMs into one identical object.

## Practical Interpretation

Use Mamba-2 as the reference when a paper claims:

- attention and SSMs are dual or closely related;
- a new SSM layer is faster due to a better scan/block algorithm;
- an architecture can be described as a structured matrix mixer;
- sequence model quality should be judged together with hardware efficiency;
- Mamba variants improve through state size, chunking, or algorithmic redesign.

The compact mental model is:

$$
\text{Mamba}
=
\text{selective SSM backbone}
$$

$$
\text{Mamba-2}
=
\text{SSD framework}
+
\text{faster Mamba-style layer}.
$$

## Limitations

- The SSD connection is structurally specific; it does not make every attention mechanism equivalent to every SSM.
- Faster core kernels do not automatically imply better end-to-end system latency for every setting.
- Language modeling competitiveness depends on scale, data, optimizer, and implementation maturity.
- Compressed recurrent-state models may still differ from attention in fine-grained retrieval behavior.
- Follow-up papers may change the preferred Mamba block design, state size, hybridization, or benchmark interpretation.

## Why It Belongs in Architecture Papers

This paper contributes both:

$$
\text{theory of sequence mixers}
$$

and:

$$
\text{a concrete Mamba-2 architecture}.
$$

It should be treated as a canonical paper for modern sequence architecture, especially when comparing Transformers, state-space models, retention, and long-convolution methods.

## Related

- [[ai/architectures|Architectures]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/retnet|RetNet]]
- [[papers/architectures/mega|Mega]]
- [[papers/architectures/hyena|Hyena]]
- [[papers/architectures/flashattention|FlashAttention]]
