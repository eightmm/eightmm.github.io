---
title: Hyena Hierarchy
aliases:
  - papers/hyena
  - papers/hyena-hierarchy
  - papers/larger-convolutional-language-models
tags:
  - papers
  - architectures
  - sequence-modeling
  - convolution
---

# Hyena Hierarchy

> The paper proposes a dense-attention-free sequence operator built from implicit long convolutions and data-controlled gating.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Hyena Hierarchy: Towards Larger Convolutional Language Models |
| Authors | Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré |
| Year | 2023 |
| Venue | ICML 2023 |
| arXiv | [2302.10866](https://arxiv.org/abs/2302.10866) |
| PMLR | [Proceedings page](https://proceedings.mlr.press/v202/poli23a.html) |
| PDF | [PMLR PDF](https://proceedings.mlr.press/v202/poli23a/poli23a.pdf) |
| Status | full note started |

## Question

Dense self-attention gives direct token-token interaction, but its sequence-length cost is quadratic:

$$
O(L^2d).
$$

Sparse, low-rank, and state-space alternatives often improve scaling but may lose some of the associative recall and reasoning behavior that makes attention useful. Hyena asks:

$$
\text{Can a dense-attention-free operator match attention-like quality at long context?}
$$

The proposed answer is:

$$
\text{implicit long convolution}
+
\text{data-controlled gating}
\Rightarrow
\text{subquadratic sequence mixing}.
$$

## Main Claim

Hyena is presented as a drop-in replacement for attention in large sequence models. Its key claim is not just lower asymptotic cost. The stronger claim is that a gated hierarchy of long convolutions can preserve enough expressivity for language modeling and long-range recall tasks:

$$
\operatorname{Hyena}(X)
\approx
\operatorname{Attention}(X)
\quad
\text{for useful long-context behavior,}
$$

while avoiding dense pairwise attention.

The architecture lesson is:

$$
\text{long convolution can be content-modulated, not only fixed local filtering.}
$$

That places Hyena between [[papers/architectures/s4|S4]], [[papers/architectures/mamba|Mamba]], and efficient-attention papers such as [[papers/architectures/longformer|Longformer]], [[papers/architectures/bigbird|BigBird]], and [[papers/architectures/performer|Performer]].

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | token sequence activations |
| Target replacement | dense self-attention operator |
| Core operator | implicit long convolution interleaved with gating |
| Mixing range | long context, potentially far beyond local CNN receptive fields |
| Cost goal | subquadratic in sequence length |
| Natural tasks in paper | long-range synthetic tasks and language modeling |
| Main comparison | attention, SSMs, sparse/low-rank/implicit sequence operators |

The abstract sequence block can be read as:

$$
X
\xrightarrow{\text{projections}}
\{x_1,\ldots,x_m\}
\xrightarrow{\text{gated long convolutions}}
Y.
$$

The important distinction from a local CNN is that the convolution kernel is long and implicitly parameterized.

## Long Convolution View

A causal 1D convolution over a sequence is:

$$
y_t
=
\sum_{\tau=0}^{t}
k_{\tau}x_{t-\tau}.
$$

If the kernel length is comparable to the context length, the operator can mix information over long ranges:

$$
k = (k_0,\ldots,k_{L-1}).
$$

Explicitly learning and storing a separate kernel value for every offset can be costly and inflexible. Hyena uses an implicit filter:

$$
k_\tau = f_\theta(\tau),
$$

where $f_\theta$ is a learned function over positions or offsets. This makes the long filter parameterized by a smaller network rather than by a raw length-$L$ vector.

## Data-Controlled Gating

Long convolution alone is still mostly fixed by position. Attention is powerful because mixing weights depend on the input. Hyena adds input-dependent gates around the convolutional operator:

$$
Y
=
G(X)
\odot
\operatorname{LongConv}_{k_\theta}(V(X)).
$$

Here:

| Symbol | Meaning |
| --- | --- |
| $V(X)$ | value-like projected sequence |
| $G(X)$ | data-controlled gate |
| $k_\theta$ | implicit long convolution filter |
| $\odot$ | elementwise modulation |

A simplified Hyena layer interleaves these pieces hierarchically:

$$
z_{i+1}
=
g_i(X)
\odot
\operatorname{LongConv}_{k_i}(z_i).
$$

The hierarchy gives multiple rounds of input-dependent modulation rather than one fixed convolution.

## Relation to Attention

Self-attention computes a pairwise content-dependent matrix:

$$
A(X)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right),
\qquad
Y=A(X)V.
$$

Hyena avoids materializing the $L\times L$ attention matrix. Its interaction pattern is not pairwise attention; it is long convolution modulated by gates:

$$
Y
=
\operatorname{GatedLongConv}(X).
$$

| Property | Dense Attention | Hyena |
| --- | --- | --- |
| token mixing | pairwise content-based weights | long convolution plus gating |
| cost driver | attention matrix $L^2$ | long convolution/filter implementation |
| context access | direct token-token paths | global filter over sequence offsets |
| content dependence | query-key scores | data-controlled gates |
| interpretability trap | attention weights may be inspected | gates and filters are less direct |

## Relation to S4 and Mamba

S4 and other structured state-space models also use a convolution view:

$$
y_t
=
\sum_{\tau=0}^{t}
K_\tau x_{t-\tau}.
$$

Hyena differs by emphasizing implicitly parameterized filters and gating hierarchy rather than a state-space discretization as the core story.

| Model | Core Route | Content Dependence |
| --- | --- | --- |
| S4 | structured state-space kernel | mostly fixed linear sequence transform |
| Hyena | implicit long convolution plus gating | data-controlled gates |
| Mamba | selective state-space scan | input-dependent SSM parameters |

This makes Hyena a useful bridge between long-convolution models and selective sequence models.

## Why It Belongs in Architecture Papers

Hyena is not just an efficiency trick. It changes the candidate set for sequence backbones:

$$
\text{attention}
\quad\text{vs}\quad
\text{sparse attention}
\quad\text{vs}\quad
\text{SSM}
\quad\text{vs}\quad
\text{gated long convolution}.
$$

For architecture reading, it adds a distinct question:

$$
\text{Can global context be modeled without explicit pairwise token scores?}
$$

That question matters for long documents, biological sequences, audio, genomics, and any setting where sequence length makes dense attention expensive.

## Evidence to Read Carefully

The paper reports results on long-range recall/reasoning tasks and language modeling. The main evidence should be read through three gates:

| Evidence | What To Check |
| --- | --- |
| long synthetic tasks | does the operator retrieve information across very long contexts? |
| language modeling | does dense-attention-free modeling approach Transformer quality? |
| speed claims | same sequence length, hardware, implementation maturity, and batch shape? |
| compute reduction | wall-clock and training budget, not only asymptotic notation |

The paper's abstract reports Transformer-quality language modeling with reduced training compute at 2K context and faster operators at longer contexts. Treat those numbers as implementation- and setting-dependent, not universal constants.

## Limits

- Hyena is not attention; tasks requiring explicit pairwise retrieval may expose different failure modes.
- Speedups depend on optimized kernels, sequence length, batch size, and hardware.
- Synthetic long-range tasks are useful probes but do not fully predict downstream utility.
- Later attention variants and SSMs change the comparison baseline.
- The operator is harder to explain than a local convolution because gating and implicit filters interact.

## What This Paper Teaches

Hyena makes sequence architecture comparison more precise:

$$
\text{sequence mixer}
\in
\{\text{attention}, \text{sparse attention}, \text{SSM}, \text{long convolution}, \text{recurrence}\}.
$$

When reading a new long-context model, ask:

- Does it materialize pairwise token interactions?
- Does it use fixed filters, learned implicit filters, or input-dependent filters?
- Can it stream autoregressively?
- Are length extrapolation and retrieval evaluated directly?
- Are throughput, memory, and quality measured at the same context length?
- Does the model rely on occasional dense attention layers?

## Concepts

- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/transformer|Transformer]]

## Related

- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/longformer|Longformer]]
- [[papers/architectures/bigbird|BigBird]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/wavenet|WaveNet]]
