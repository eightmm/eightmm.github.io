---
title: Gated Linear Attention
aliases:
  - papers/gated-linear-attention
  - papers/gla
  - papers/flash-linear-attention
  - papers/hardware-efficient-linear-attention
tags:
  - papers
  - architectures
  - transformer
  - attention
  - efficient-attention
  - linear-attention
  - sequence-modeling
---

# Gated Linear Attention

> Gated Linear Attention is important because it revisits linear attention after FlashAttention and Mamba: the paper treats the attention formula, recurrent inference form, and GPU memory movement as one architecture problem.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Gated Linear Attention Transformers with Hardware-Efficient Training |
| Authors | Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim |
| Year | 2024 |
| Venue | ICML 2024 |
| PMLR | [PMLR 235:56501-56523](https://proceedings.mlr.press/v235/yang24ab.html) |
| arXiv | [2312.06635](https://arxiv.org/abs/2312.06635) |
| OpenReview | [ia5XvxFUJT](https://openreview.net/forum?id=ia5XvxFUJT) |
| Status | full note started |

## Question

Softmax attention gives strong content-based retrieval:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

But it forms a token-token matrix:

$$
QK^\top
\in
\mathbb{R}^{T\times T},
$$

which creates quadratic sequence cost. Linear attention tries to avoid this by replacing the softmax kernel with feature maps:

$$
\operatorname{softmax}(q^\top k)
\approx
\phi(q)^\top \phi(k).
$$

Then the computation can be rearranged:

$$
\Phi(Q)(\Phi(K)^\top V),
$$

without materializing the full $T\times T$ attention matrix.

The problem is that earlier linear attention often had two practical weaknesses:

| Problem | Why It Matters |
| --- | --- |
| quality gap | softmax attention remained stronger on language modeling |
| hardware gap | linear asymptotic complexity did not automatically mean faster wall-clock training |

GLA asks:

$$
\text{Can linear attention be made both more expressive and hardware-efficient?}
$$

## Main Claim

The paper makes two connected architecture claims:

$$
\text{linear attention}
\xrightarrow{\text{chunkwise hardware-aware algorithm}}
\text{FlashLinearAttention}
$$

and:

$$
\text{linear attention}
\xrightarrow{\text{data-dependent gates}}
\text{Gated Linear Attention}.
$$

The durable idea is:

$$
\text{efficient sequence architecture}
=
\text{mathematical form}
+
\text{recurrent form}
+
\text{hardware-aware parallel form}.
$$

For this wiki, GLA belongs between [[papers/architectures/performer|Performer]], [[papers/architectures/mega|Mega]], [[papers/architectures/retnet|RetNet]], [[papers/architectures/mamba|Mamba]], and [[papers/architectures/flashattention|FlashAttention]].

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Baseline | Transformer attention layer |
| Replacement | gated linear attention layer |
| Main state | matrix-valued recurrent hidden state |
| Gate | data-dependent forget gate over recurrent state |
| Training target | parallel/chunkwise hardware-efficient algorithm |
| Inference target | linear-time recurrent decoding |
| Main comparison | softmax attention, RetNet, Mamba, FlashAttention-style kernels |

The language-model objective is unchanged:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
$$

The architecture change is in the sequence mixer.

## Linear Attention Recurrence

Unnormalized linear attention can be written with a recurrent state:

$$
S_t
=
S_{t-1}
+
k_t^\top v_t,
$$

$$
o_t
=
q_t S_t.
$$

Here $S_t$ is matrix-valued. If:

$$
k_t\in\mathbb{R}^{d_k},
\qquad
v_t\in\mathbb{R}^{d_v},
$$

then:

$$
S_t\in\mathbb{R}^{d_k\times d_v}.
$$

This recurrence gives linear-time inference because the past is compressed into $S_t$ rather than stored as all previous keys and values.

The tradeoff is compression:

$$
\{(k_i,v_i)\}_{i\le t}
\rightarrow
S_t.
$$

If the compressed state cannot preserve the needed retrieval information, quality can drop relative to softmax attention.

## Gating

GLA adds data-dependent forgetting. A simple way to read the update is:

$$
S_t
=
g_t \odot S_{t-1}
+
k_t^\top v_t,
$$

where $g_t$ is an input-dependent gate and $\odot$ is elementwise multiplication or a structured gating operation depending on the exact implementation.

The gate gives the model a learned way to control memory:

| Gate Behavior | Effect |
| --- | --- |
| $g_t\approx 1$ | keep previous state |
| $g_t\approx 0$ | forget previous state |
| feature-dependent $g_t$ | forget some dimensions more than others |

This is why GLA belongs near [[concepts/architectures/gating|Gating]] and [[concepts/architectures/state-space-model|state-space models]], not only near classic attention.

## Parallel, Recurrent, and Chunkwise Forms

The paper emphasizes that the same layer should be read in several equivalent or related forms.

| Form | Use |
| --- | --- |
| parallel form | training over many sequence positions |
| recurrent form | autoregressive inference |
| chunkwise form | hardware-aware training and memory movement control |

This mirrors a broader pattern in modern sequence architecture:

$$
\text{paper quality}
\neq
\text{only asymptotic complexity}.
$$

A layer with $O(T)$ arithmetic can still be slow if it moves too much data through high-bandwidth memory or fails to use tensor cores effectively.

## Hardware-Aware Lesson

FlashAttention showed that exact softmax attention can become much faster by reducing memory traffic and tiling the computation.

GLA applies a similar lesson to linear attention:

$$
\text{linear attention}
\text{ should be optimized for IO, parallelism, and chunk structure}.
$$

The paper reports FlashLinearAttention as a hardware-efficient linear attention implementation and then generalizes it to the gated case.

For architecture reading, the important distinction is:

| Claim Type | Evidence Needed |
| --- | --- |
| asymptotic complexity | formulas in $T,d,H,B$ |
| kernel efficiency | wall-clock throughput under hardware, dtype, batch, and sequence settings |
| model quality | perplexity, downstream benchmark, and length generalization |
| inference benefit | recurrent decoding state size and latency |

Do not read "linear attention" as automatically faster than FlashAttention. The implementation path matters.

## Relation to Softmax Attention

Softmax attention keeps an explicit distribution over previous tokens:

$$
p_{t,i}
=
\operatorname{softmax}_i(q_t^\top k_i).
$$

GLA compresses the prefix into recurrent state. This changes the retrieval contract:

| Family | Past Representation | Strength | Risk |
| --- | --- | --- | --- |
| softmax attention | explicit KV cache | precise token-token retrieval | quadratic training memory; large KV cache |
| linear attention | recurrent matrix state | linear recurrent inference | compressed-state bottleneck |
| GLA | gated recurrent matrix state | adaptive forgetting and better quality | still may struggle with recall-heavy tasks |

This makes GLA a sequence mixer alternative, not a drop-in semantic equivalent to softmax attention.

## Relation to RetNet and Mamba

GLA sits in the same design pressure as [[papers/architectures/retnet|RetNet]] and [[papers/architectures/mamba|Mamba]]:

$$
\text{Transformer-like training}
\quad+\quad
\text{RNN-like inference}
\quad+\quad
\text{long-context efficiency}.
$$

| Paper | Sequence State |
| --- | --- |
| [RetNet](/papers/architectures/retnet) | retention state with decay |
| [Mamba](/papers/architectures/mamba) | selective SSM state |
| [Mamba-2](/papers/architectures/mamba-2) | structured state-space dual matrix view |
| [GLA](/papers/architectures/gated-linear-attention) | gated linear-attention matrix state |

The useful comparison is not "which one replaces Transformers." The useful comparison is:

$$
\text{what information is stored in recurrent state, and how is it updated?}
$$

## Evidence to Read Carefully

The paper reports competitive moderate-scale language modeling, length generalization, and training throughput relative to efficient sequence baselines.

Read each evidence type separately:

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| perplexity | GLA can be competitive at reported scale | universal replacement for softmax attention |
| length generalization | trained-short tested-long behavior can be strong | robust retrieval on every long-context task |
| throughput | implementation is hardware-aware | architecture-only speed independent of kernels |
| comparison to Mamba/RetNet | GLA is in the same modern sequence family | dominance under all parameter/data budgets |
| standalone layer speed | kernel can be efficient | full model serving is automatically faster |

The strongest durable lesson is the combination of gating and hardware-aware linear attention.

## Practical Checks

- Identify whether a paper uses softmax attention, kernel linear attention, gated linear attention, retention, or SSM.
- Track recurrent state size: vector state, matrix state, KV cache, or structured SSM state.
- Separate training parallelism from inference recurrence.
- Compare wall-clock speed under the same dtype, hardware, sequence length, batch size, and implementation.
- Check whether long-context results test recall, summarization, perplexity, or synthetic extrapolation.
- Verify whether "linear" means arithmetic complexity, memory complexity, or actual measured throughput.

## Where It Fits

GLA fills a missing slot in the architecture shelf:

$$
\text{Performer/Linformer}
\rightarrow
\text{efficient attention approximations}
\rightarrow
\text{GLA}
\rightarrow
\text{modern recurrent-linear sequence backbones}.
$$

It is especially useful before reading newer linear-attention and Transformer-to-RNN papers such as delta-rule variants, gated slot attention, and hybrid SSM/attention models.

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/linformer|Linformer]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
- [[papers/architectures/mega|Mega]]
- [[papers/architectures/retnet|RetNet]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/mamba-2|Mamba-2]]
- [[papers/architectures/jamba|Jamba]]
