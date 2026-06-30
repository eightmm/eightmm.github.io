---
title: RetNet
aliases:
  - papers/retnet
  - papers/retentive-network
  - papers/retentive-network-successor-transformer
tags:
  - papers
  - architectures
  - retention
  - recurrent
  - sequence-modeling
  - language-model
---

# RetNet

> RetNet proposes retention as a sequence mixing mechanism with three compatible computation modes: parallel training, recurrent inference, and chunkwise recurrent long-sequence processing.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Retentive Network: A Successor to Transformer for Large Language Models |
| Authors | Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei |
| Year | 2023 |
| Venue | arXiv preprint |
| arXiv | [2307.08621](https://arxiv.org/abs/2307.08621) |
| Microsoft Research | [Publication page](https://www.microsoft.com/en-us/research/publication/retentive-network-a-successor-to-transformer-for-large-language-models/) |
| Status | full note started |

## Question

The Transformer solved a training problem of earlier recurrent models: it lets all tokens in a sequence be processed in parallel. The cost is that causal decoding stores a growing key-value cache:

$$
\text{KV cache}
\propto
L\cdot n_{\mathrm{layers}}\cdot d.
$$

Classic RNNs have cheaper streaming inference:

$$
s_t = F_\theta(s_{t-1}, x_t)
$$

but they historically gave up the training parallelism and scaling behavior that made Transformers dominant.

RetNet asks:

$$
\text{Can a language-model backbone keep Transformer-style training parallelism}
\quad
\text{and}
\quad
\text{RNN-style inference state?}
$$

The proposed answer is a retention mechanism with three equivalent or compatible forms:

| Mode | Use |
| --- | --- |
| parallel retention | train over tokens in parallel |
| recurrent retention | decode with a compact recurrent state |
| chunkwise recurrent retention | encode local chunks in parallel while passing global state across chunks |

## Main Claim

The durable architecture claim is:

$$
\text{retention}
\approx
\text{attention-like sequence mixing with recurrent state reuse}.
$$

RetNet does not only propose another sparse attention pattern. It changes the sequence mixer from explicit softmax attention:

$$
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

to retention, where query-key interactions are modulated by causal decay and can be rewritten as a recurrent update.

In simplified form:

$$
Y
=
\left(QK^\top \odot D\right)V
$$

where $D$ is a causal decay matrix. The recurrent version keeps a running key-value state:

$$
S_t
=
\gamma S_{t-1} + k_t^\top v_t
$$

$$
y_t
=
q_t S_t.
$$

The important point is not the exact notation. It is the contract:

$$
\text{parallel matrix form}
\quad\leftrightarrow\quad
\text{recurrent state form}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Output | contextual token representations or next-token logits |
| Target replacement | multi-head self-attention in decoder-style language models |
| Sequence mixer | multi-scale retention |
| Training form | parallel or chunkwise recurrent |
| Inference form | recurrent retention state |
| Inference memory target | constant or nearly constant state rather than growing KV cache |
| Natural comparison | decoder-only Transformer, efficient attention, RWKV, SSM-style sequence models |

The block scaffold remains close to a modern Transformer:

$$
X
\xrightarrow{\operatorname{LayerNorm}}
\operatorname{MultiScaleRetention}
\xrightarrow{\text{residual}}
\operatorname{FFN}
\xrightarrow{\text{residual}}
Y.
$$

This is why RetNet belongs in the same reading route as [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], [[papers/architectures/rwkv|RWKV]], [[papers/architectures/s4|S4]], [[papers/architectures/hyena|Hyena]], and [[papers/architectures/mamba|Mamba]].

## Retention Mechanism

Retention can be read as attention without softmax normalization and with a causal decay structure.

For token activations:

$$
X\in\mathbb{R}^{L\times d},
$$

project:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

Parallel retention has the shape:

$$
\operatorname{Ret}(X)
=
\left(QK^\top \odot D\right)V.
$$

The decay matrix $D$ enforces two things:

| Property | Meaning |
| --- | --- |
| causal mask | position $t$ cannot use future positions |
| exponential decay | older tokens are discounted by distance |

A useful simplified decay is:

$$
D_{t,i}
=
\begin{cases}
\gamma^{t-i}, & i\le t \\
0, & i>t
\end{cases}
$$

where $0<\gamma<1$ controls the retention horizon.

## Recurrent Form

The same computation can be written as a recurrent state update:

$$
S_t
=
\gamma S_{t-1}
+
k_t^\top v_t
$$

$$
y_t
=
q_t S_t.
$$

This makes autoregressive decoding closer to an RNN:

$$
(S_{t-1}, x_t)
\rightarrow
(S_t, y_t).
$$

The inference state stores accumulated key-value information instead of all past token keys and values.

| Family | Stored During Decoding |
| --- | --- |
| Transformer | previous $K,V$ for each layer and token |
| RetNet | recurrent retention state $S_t$ |
| RWKV | recurrent weighted key-value state |
| SSM/Mamba | scan state or selective state |

This is the practical reason RetNet is important for deployment-oriented architecture reading.

## Chunkwise Recurrent Form

The chunkwise form is the bridge between fully parallel training and fully recurrent inference.

Split a long sequence into chunks:

$$
X = [C_1, C_2, \ldots, C_m].
$$

Within a chunk, compute retention in parallel:

$$
C_j
\xrightarrow{\text{parallel retention}}
Y_j.
$$

Across chunks, pass a recurrent state:

$$
S_j
=
F(S_{j-1}, C_j).
$$

This gives a useful long-sequence pattern:

$$
\text{local parallelism}
+
\text{global recurrence}.
$$

That is a different tradeoff from sparse attention windows, FlashAttention, or standard KV-cache decoding.

## Multi-Scale Retention

RetNet uses multiple retention heads with different decay rates. The idea is similar to using several memory horizons:

| Head Type | Intuition |
| --- | --- |
| fast decay | local or short-range information |
| medium decay | phrase or segment-level memory |
| slow decay | longer-range state |

With heads $h=1,\ldots,H$:

$$
D^{(h)}_{t,i}
=
\gamma_h^{t-i}
\mathbf{1}[i\le t].
$$

Then each head computes:

$$
Y^{(h)}
=
\left(Q^{(h)}K^{(h)\top}\odot D^{(h)}\right)V^{(h)}.
$$

The outputs are normalized and combined. This makes retention less like a single fixed recurrence and more like a bank of decaying memory traces.

## Relation to Attention

Self-attention computes:

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right),
\qquad
Y=AV.
$$

Retention computes a causal decayed interaction:

$$
R
=
QK^\top\odot D,
\qquad
Y=RV.
$$

The biggest conceptual differences:

| Axis | Transformer Attention | Retention |
| --- | --- | --- |
| interaction | pairwise token-token scores | pairwise form with decay, plus recurrent rewrite |
| normalization | softmax over accessible tokens | retention-specific normalization/stabilization |
| inference cache | growing KV cache | recurrent state |
| long sequence path | sparse/linear attention, FlashAttention, chunking, retrieval | chunkwise recurrence |
| inductive bias | direct content addressing | decayed memory with multiple horizons |

RetNet therefore should be read as a sequence architecture, not just an implementation optimization.

## Relation to RWKV, S4, Hyena, and Mamba

| Paper | Core Sequence Mixer | Why Compare |
| --- | --- | --- |
| [RWKV](/papers/architectures/rwkv) | recurrent weighted key-value update | also aims for Transformer-like training and RNN-like inference |
| [S4](/papers/architectures/s4) | structured state-space sequence layer | efficient long-range state dynamics |
| [Hyena](/papers/architectures/hyena) | gated implicit long convolution | dense-attention-free long-context mixing |
| [Mamba](/papers/architectures/mamba) | selective state-space scan | input-dependent recurrent sequence state |
| [RetNet](/papers/architectures/retnet) | multi-scale retention | parallel, recurrent, and chunkwise recurrent forms |

RetNet is closest to the question:

$$
\text{Can the attention matrix be replaced by a retention operator that still scales like a language-model backbone?}
$$

RWKV is closest to:

$$
\text{Can an RNN-like model be trained at Transformer-era scale?}
$$

Mamba is closest to:

$$
\text{Can a selective state-space scan become a general sequence backbone?}
$$

These papers should be read together rather than as isolated "Transformer killers".

## Evidence to Read Carefully

The paper reports language modeling, scaling behavior, training memory/speed, and inference efficiency. The evidence should be read through separate questions.

| Evidence | What It Supports | What It Does Not Prove Alone |
| --- | --- | --- |
| language-model scaling curves | RetNet can compete with Transformer-style baselines under reported settings | universal dominance over attention |
| recurrent inference measurements | lower memory and latency from avoiding growing KV cache | better reasoning or retrieval at all context lengths |
| chunkwise training results | long-sequence training can exploit local parallelism and global recurrence | that chunking policy is optimal for every workload |
| ablations on decay/gating/multi-scale choices | retention design details matter | exact component importance in all model sizes |
| comparison with FlashAttention | RetNet has a different memory/computation path | that kernel engineering differences are fully isolated |

The key audit is:

$$
\text{architecture gain}
\neq
\text{only implementation or benchmark-budget gain}.
$$

## Practical Interpretation

Use RetNet as a reference when a paper claims:

- constant-state or low-cost autoregressive decoding;
- no KV cache or reduced KV-cache pressure;
- Transformer-comparable language modeling with recurrence;
- chunkwise sequence modeling;
- long-context efficiency without dense attention.

Do not use RetNet as a generic proof that attention is obsolete. It is a concrete architecture point in a broader design space:

$$
\text{attention}
\leftrightarrow
\text{linear attention}
\leftrightarrow
\text{retention}
\leftrightarrow
\text{recurrent state}
\leftrightarrow
\text{state-space scan}.
$$

## Implementation Checks

When reading RetNet implementations or follow-up papers, check:

| Check | Why It Matters |
| --- | --- |
| recurrent and parallel equivalence | claimed training/inference forms should compute compatible outputs |
| decay parameterization | retention horizon depends on $\gamma$ choices |
| normalization | retention scores can need stabilization |
| chunk policy | chunk size affects memory, latency, and quality |
| causal masking | leakage can appear if parallel form is implemented incorrectly |
| comparison baseline | Transformer baseline should match parameter count, data, training tokens, and kernel budget |

## Why It Belongs in Architecture Papers

RetNet contributes a reusable sequence mixing primitive. It is not mainly a benchmark paper, dataset paper, or application paper.

Its durable architectural idea is:

$$
\text{retention state}
=
\text{decayed key-value memory that supports multiple compute views}.
$$

That makes it a canonical architecture note for the sequence-model shelf.

## Related

- [[ai/architectures|Architectures]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/gating|Gating]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/rwkv|RWKV]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/hyena|Hyena]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/flashattention|FlashAttention]]
