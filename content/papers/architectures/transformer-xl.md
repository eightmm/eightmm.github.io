---
title: Transformer-XL
aliases:
  - papers/transformer-xl
  - papers/architectures/attentive-language-models-beyond-fixed-length-context
  - papers/attentive-language-models-beyond-fixed-length-context
tags:
  - papers
  - architectures
  - transformer
  - sequence-modeling
---

# Transformer-XL

> The paper extends Transformer language models beyond fixed-length segments using segment-level recurrence and relative positional encoding.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Transformer-XL: Attentive Language Models beyond a Fixed-Length Context |
| Authors | Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov |
| Year | 2019 |
| Venue | ACL 2019 |
| arXiv | [1901.02860](https://arxiv.org/abs/1901.02860) |
| ACL Anthology | [P19-1285](https://aclanthology.org/P19-1285/) |
| Code | [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl) |
| Status | full note started |

## One-Line Takeaway

Transformer-XL changes a fixed-window Transformer language model into a recurrent segment model:

$$
\text{current segment}
+
\text{cached previous hidden states}
\rightarrow
\text{longer effective context}.
$$

The two durable architecture ideas are:

1. reuse previous segment hidden states as memory;
2. use relative position information so cached states remain meaningful when reused.

## Question

A vanilla autoregressive Transformer usually trains on fixed-length chunks:

$$
x_{1:L},\quad x_{L+1:2L},\quad x_{2L+1:3L},\ldots
$$

Within a chunk, causal attention can see earlier tokens:

$$
h_t
=
f(x_{\le t})
\quad
\text{for }t\in[1,L].
$$

But at the next chunk, the model starts over:

$$
h_{L+1}
\not\leftarrow
h_{\le L}.
$$

This creates two problems:

| Problem | Meaning |
| --- | --- |
| fixed context | dependencies beyond the segment length are unavailable |
| context fragmentation | training segments cut across natural semantic boundaries |

The architecture question is:

$$
\text{Can a Transformer keep useful memory across segments without recomputing the full prefix?}
$$

## Main Claim

Segment-level recurrence lets each segment attend to cached hidden states from previous segments. Relative positional encoding makes that memory reusable across segment boundaries.

For layer $n$ and segment $\tau$, let:

$$
H_\tau^n
\in
\mathbb{R}^{L\times d}
$$

be hidden states for the current segment, and:

$$
\operatorname{SG}(H_{\tau-1}^n)
$$

be the previous segment hidden states with stopped gradients.

Transformer-XL builds an extended context:

$$
\tilde{H}_\tau^{n-1}
=
\left[
\operatorname{SG}(H_{\tau-1}^{n-1})
\circ
H_\tau^{n-1}
\right],
$$

where $\circ$ denotes concatenation along the sequence dimension.

Then the current segment queries attend over previous memory plus current states:

$$
Q_\tau^n
=
H_\tau^{n-1}W_q^n,
$$

$$
K_\tau^n
=
\tilde{H}_\tau^{n-1}W_k^n,
\qquad
V_\tau^n
=
\tilde{H}_\tau^{n-1}W_v^n.
$$

The current segment output remains length $L$, but its attention context is longer than $L$.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | autoregressive token sequence split into fixed-length segments |
| Segment length | current train/eval chunk length $L$ |
| Memory length | number of previous hidden states cached per layer |
| Query source | current segment states only |
| Key/value source | previous memory plus current segment states |
| Gradient through memory | stopped at previous segment cache |
| Position scheme | relative positional encoding |
| Output | next-token distribution for current segment |
| Main benefit | longer effective context and faster evaluation via reuse |

Transformer-XL is not merely a bigger context window. It is a recurrent Transformer over hidden-state segments.

## Segment-Level Recurrence

Let segment $\tau$ contain tokens:

$$
s_\tau
=
(x_{\tau,1},\ldots,x_{\tau,L}).
$$

For each layer $n$, Transformer-XL keeps a memory:

$$
M_\tau^n
=
\operatorname{SG}(H_{\tau-1}^n).
$$

The attention context for the current layer is:

$$
\tilde{H}_\tau^{n-1}
=
[M_\tau^{n-1}; H_\tau^{n-1}].
$$

Then self-attention is computed with:

$$
\operatorname{Attn}
\left(
H_\tau^{n-1},
\tilde{H}_\tau^{n-1}
\right)
=
\operatorname{softmax}
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
+
\operatorname{mask}
\right)V.
$$

The causal mask allows each current token to attend to:

- earlier cached memory states;
- earlier positions in the current segment;
- itself, depending on the implementation convention.

It does not allow attention to future current-segment positions.

## Why Stop Gradient

The memory is reused for context but not backpropagated through indefinitely:

$$
M_\tau^n
=
\operatorname{SG}(H_{\tau-1}^n).
$$

This keeps training tractable:

$$
\text{context length can grow}
\quad
\text{without}
\quad
\text{unbounded backpropagation through time}.
$$

The tradeoff is clear:

| Choice | Benefit | Cost |
| --- | --- | --- |
| stop-gradient memory | stable and efficient training | no gradient credit assignment through all past segments |
| full prefix recomputation | exact full-context gradients | expensive and fixed by maximum context length |
| no memory | simple parallel chunks | context fragmentation |

## Relative Positional Encoding

If a model uses absolute position embeddings:

$$
e_i = w_i + p_i,
$$

cached hidden states from a previous segment carry positions tied to their old segment. Reusing them in a new segment can create positional confusion.

Transformer-XL instead uses relative position information in attention. The attention logit between query position $i$ and key position $j$ depends on the relative distance:

$$
i-j.
$$

A simplified relative attention score is:

$$
A_{i,j}
=
q_i^\top k_j
+
q_i^\top r_{i-j},
$$

where:

| Symbol | Meaning |
| --- | --- |
| $q_i$ | query vector at current position $i$ |
| $k_j$ | content key vector at memory/current position $j$ |
| $r_{i-j}$ | relative positional embedding for offset $i-j$ |

The paper uses a decomposed formulation with content and position terms:

$$
A_{i,j}
=
q_i^\top k_j
+
q_i^\top r_{i-j}
+
u^\top k_j
+
v^\top r_{i-j}.
$$

Here $u$ and $v$ are learned global content and positional biases.

The durable idea:

$$
\text{memory reuse}
\Rightarrow
\text{positions must be relative to the current query}.
$$

## Attention With Memory

For one layer, current queries:

$$
Q
=
H_\tau W_q
\in
\mathbb{R}^{L\times d_k}
$$

attend to keys from memory plus current segment:

$$
K
=
[M_\tau;H_\tau]W_k
\in
\mathbb{R}^{(M+L)\times d_k}.
$$

The attention matrix has shape:

$$
A
\in
\mathbb{R}^{L\times(M+L)}.
$$

So complexity per segment is:

$$
O(L(M+L)d)
$$

instead of:

$$
O((M+L)^2d)
$$

for recomputing all outputs over the full concatenated prefix.

This is why Transformer-XL can be much faster at evaluation: cached hidden states avoid repeated prefix computation.

## Context Fragmentation

In fixed-length training, a dependency crossing a segment boundary is invisible:

$$
x_{L-k}
\rightarrow
x_{L+m}
$$

if the model only sees the current chunk.

Segment memory changes the training signal:

$$
p(x_{\tau,t}\mid x_{\tau,<t}, M_\tau)
$$

instead of:

$$
p(x_{\tau,t}\mid x_{\tau,<t}).
$$

This gives the model a better approximation to document-level continuity even when training still uses fixed-size segments.

## Relation To Vanilla Transformer

| Axis | Vanilla Transformer LM | Transformer-XL |
| --- | --- | --- |
| context | fixed current segment | current segment plus memory |
| recurrence | none across segments | hidden-state recurrence |
| position | often absolute or fixed within segment | relative position for reuse |
| evaluation | recompute prefix or sliding window | cache previous hidden states |
| training gradient | within segment | within segment, memory stop-gradient |
| failure mode | context fragmentation | stale/compressed memory and finite cache |

Transformer-XL does not remove attention's quadratic behavior inside the attended span. It changes how the attended span is maintained across time.

## Relation To RNNs, Mamba, And Long-Context Models

Transformer-XL reintroduces recurrence into a Transformer:

$$
M_{\tau+1}=H_\tau.
$$

This looks RNN-like at the segment level, but within each segment it still uses parallel self-attention.

| Model Family | Memory Form |
| --- | --- |
| [[concepts/architectures/rnn|RNN]] / [[concepts/architectures/lstm|LSTM]] | fixed-size recurrent hidden state |
| Transformer-XL | cached previous hidden states |
| [[concepts/architectures/state-space-model|State-space model]] / [[concepts/architectures/mamba|Mamba]] | structured state update over sequence |
| Retrieval-augmented models | external retrieved text or vectors |
| Long-context Transformers | larger direct attention window or sparse attention |

The important distinction:

$$
\text{Transformer-XL memory}
=
\text{past hidden states used as attention keys/values}.
$$

It is not a learned compressed summary unless the hidden states themselves become summary-like through training.

## Evidence Pattern

The paper supports the architecture with:

| Evidence | What It Supports |
| --- | --- |
| language modeling benchmarks | lower bpc/perplexity across word-level and character-level datasets |
| ablations | recurrence and relative position both matter |
| relative effective context length | model uses longer dependencies than baselines |
| evaluation speed | cached memory avoids repeated computation |
| long generation examples | segment recurrence improves continuity over thousands of tokens |

The strongest architecture evidence is the combination of ablation and effective-context analysis. Benchmark gains alone could come from tuning or compute, but the recurrence/relative-position controls make the architectural mechanism more credible.

## Practical Reading For Modern LLMs

Modern LLMs often use KV caches at inference:

$$
K_{\le t},V_{\le t}
$$

so Transformer-XL may feel familiar. The difference is the training and segmentation setup.

Transformer-XL explicitly trains a model to reuse previous segment hidden states with a relative positional scheme. Standard decoder-only LLM inference cache is often a direct consequence of causal attention over the full context used at training.

Reading questions:

| Question | Why |
| --- | --- |
| Was memory used during training or only inference? | train/test context mismatch changes behavior |
| Are positions absolute, relative, rotary, or biased? | context extrapolation depends on position scheme |
| What is cached: hidden states, keys/values, summaries, or retrieved text? | memory semantics differ |
| Is gradient stopped through memory? | affects long-range credit assignment |
| Is evaluation length longer than training length? | extrapolation claim needs direct evidence |

## Why It Belongs In Architecture Papers

Transformer-XL is a canonical architecture note because it creates a clean design axis:

$$
\text{attention}
+
\text{recurrence}
+
\text{relative position}
\rightarrow
\text{longer effective context}.
$$

This axis is still relevant when reading:

- long-context LLMs;
- recurrent memory Transformers;
- state-space sequence models;
- retrieval and memory-augmented agents;
- protein or genome sequence models with long-range dependencies.

Even when a modern model does not use Transformer-XL directly, the note gives a vocabulary for separating:

| Claim Type | Check |
| --- | --- |
| longer context | effective context length, not only max token count |
| faster inference | cached computation and memory bandwidth |
| better continuity | boundary-crossing tasks or long generation |
| better position handling | ablation over positional scheme |
| better memory | what is stored, how long, and whether it is train-time visible |

## Limitations

- Memory length is still finite.
- Cached hidden states are detached, so long-range gradient flow is limited.
- More memory increases attention cost and memory bandwidth.
- Relative position helps reuse, but does not guarantee arbitrary-length extrapolation.
- It was designed for language modeling; applying the idea to other modalities requires a clear segment boundary and memory contract.
- Cached hidden states may be stale if later layers would have changed them under full-context recomputation.

The concise limitation:

$$
\text{Transformer-XL extends context}
\neq
\text{unbounded exact full-context attention}.
$$

## What To Remember

- Fixed-length Transformer LMs suffer from context fragmentation.
- Transformer-XL caches previous segment hidden states as memory.
- Current queries attend to memory plus current segment keys/values.
- Gradients through memory are stopped for tractable training.
- Relative positional encoding is necessary because cached states cross segment boundaries.
- The paper is a key bridge between Transformers, recurrent memory, KV caching, and long-context architectures.

## Links

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/llama|LLaMA]]
- [[papers/architectures/mamba|Mamba]]
