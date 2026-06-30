---
title: Linformer
aliases:
  - papers/linformer
  - papers/self-attention-with-linear-complexity
tags:
  - papers
  - architectures
  - transformer
  - attention
  - efficient-attention
  - low-rank
---

# Linformer

> Linformer makes self-attention linear in sequence length by projecting keys and values along the sequence dimension, based on the claim that the attention matrix can be approximated as low-rank.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Linformer: Self-Attention with Linear Complexity |
| Authors | Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma |
| Year | 2020 |
| Venue | arXiv preprint |
| arXiv | [2006.04768](https://arxiv.org/abs/2006.04768) |
| Status | seed note |

## One-Line Takeaway

Linformer is the canonical low-rank attention paper in this wiki:

$$
\text{dense } n\times n \text{ attention}
\rightarrow
\text{project keys and values to } k \text{ positions}
\rightarrow
O(nk) \text{ attention}.
$$

Keep it separate from sparse attention, LSH attention, random-feature attention, and exact attention kernels.

## Question

Standard self-attention computes:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

For a sequence length $n$:

$$
Q,K,V\in\mathbb{R}^{n\times d},
\qquad
QK^\top\in\mathbb{R}^{n\times n}.
$$

The quadratic object is the attention score or attention probability matrix:

$$
P
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right).
$$

The paper asks:

$$
\text{Can } P\in\mathbb{R}^{n\times n}
\text{ be represented well enough through a lower-rank approximation?}
$$

If yes, full attention does not need every token to keep an independent key/value slot.

## Main Claim

Linformer claims that self-attention can often be approximated by a low-rank matrix. The architecture then reduces sequence length inside attention by learned projection matrices.

Let:

$$
E,F\in\mathbb{R}^{k\times n},
\qquad
k\ll n.
$$

Instead of attending to all $n$ key and value positions, Linformer computes:

$$
\tilde K = EK,
\qquad
\tilde V = FV,
$$

where:

$$
\tilde K,\tilde V\in\mathbb{R}^{k\times d}.
$$

Attention becomes:

$$
\operatorname{LinformerAttention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{Q\tilde K^\top}{\sqrt{d}}
\right)
\tilde V.
$$

The score matrix shape changes from:

$$
n\times n
\quad\text{to}\quad
n\times k.
$$

If $k$ is fixed or grows slowly compared with $n$, the attention cost is linear in $n$.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | sequence representations $X\in\mathbb{R}^{n\times d_{\text{model}}}$ |
| Baseline block | Transformer self-attention |
| Changed part | key/value sequence dimension is compressed before attention |
| Main assumption | attention matrix is effectively low-rank |
| Projection size | $k$, with $k\ll n$ |
| Cost target | $O(nk)$ instead of $O(n^2)$ attention cost |
| Main risk | useful information can be lost if $k$ is too small or the low-rank assumption fails |

Linformer is an attention replacement. It is not a new tokenizer, language-model objective, or pretraining recipe.

## Shape Walkthrough

Start with hidden states:

$$
X\in\mathbb{R}^{n\times d_{\text{model}}}.
$$

Linear projections produce:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

with:

$$
Q,K,V\in\mathbb{R}^{n\times d_h}.
$$

Dense attention builds:

$$
S=QK^\top\in\mathbb{R}^{n\times n}.
$$

Linformer first compresses the token axis:

$$
\tilde K=EK,\qquad \tilde V=FV,
$$

with:

$$
E,F\in\mathbb{R}^{k\times n},
\qquad
\tilde K,\tilde V\in\mathbb{R}^{k\times d_h}.
$$

The attention score becomes:

$$
\tilde S=Q\tilde K^\top\in\mathbb{R}^{n\times k}.
$$

Then:

$$
Y
=
\operatorname{softmax}
\left(
\frac{\tilde S}{\sqrt{d_h}}
\right)
\tilde V,
\qquad
Y\in\mathbb{R}^{n\times d_h}.
$$

The output still has one vector per input token. Only the key/value memory being attended over is compressed.

## Complexity

Dense self-attention:

$$
O(n^2d_h)
$$

for score computation, plus:

$$
O(n^2)
$$

for the attention matrix.

Linformer attention:

$$
O(nkd_h)
$$

after projecting keys and values to length $k$.

The projection itself costs:

$$
O(nkd_h)
$$

if implemented as a dense sequence projection. The intended regime is:

$$
k\ll n.
$$

So the useful comparison is:

| Attention | Score Shape | Sequence Cost |
| --- | --- | --- |
| dense attention | $n\times n$ | $O(n^2)$ |
| Linformer | $n\times k$ | $O(nk)$ |

Calling this "linear" assumes $k$ is treated as a chosen projection dimension rather than another value that scales like $n$.

## Low-Rank View

The attention probability matrix is:

$$
P
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right).
$$

A low-rank approximation would mean:

$$
P
\approx
\hat P,
\qquad
\operatorname{rank}(\hat P)\le k.
$$

Linformer does not explicitly compute an SVD of $P$. Instead, it learns projections that reduce the key/value sequence axis before the attention matrix is formed.

The architecture bet is:

$$
\text{many attention maps do not need full rank } n.
$$

If that bet is true, a compressed key/value memory can preserve enough behavior.

## Relation to Other Efficient Attention Papers

| Paper | Route | What Changes |
| --- | --- | --- |
| [Longformer](/papers/architectures/longformer) | sparse local/global attention | attention graph is hand-designed |
| [BigBird](/papers/architectures/bigbird) | local/global/random sparse attention | sparse graph adds random and global connectivity |
| [Reformer](/papers/architectures/reformer) | LSH attention | candidate keys are selected by approximate similarity buckets |
| Linformer | low-rank projection | keys and values are compressed along sequence length |
| [Performer](/papers/architectures/performer) | random-feature kernel attention | softmax attention is approximated algebraically |
| [FlashAttention](/papers/architectures/flashattention) | IO-aware exact attention | dense attention remains exact but is tiled efficiently |

The important distinction:

$$
\text{efficient attention}
\ne
\text{one method family}.
$$

Sparse attention changes connectivity. Linformer changes rank. Performer changes kernel approximation. FlashAttention changes the implementation schedule for exact dense attention.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| attention maps can be low-rank | empirical spectrum analysis | motivates sequence-length projection | rank can depend on layer, task, model, and input distribution |
| linear attention reduces memory/time | complexity analysis and runtime experiments | long inputs can become cheaper | projection overhead and implementation constants matter |
| performance stays close to Transformer | benchmark comparisons | compression can preserve useful behavior | the best $k$ is a hyperparameter and may not transfer |
| sharing projection matrices is possible | ablation/config variants | memory can be reduced further | sharing may reduce flexibility |

Read the paper as a claim about the structure of attention matrices, not as a universal replacement for full attention.

## Implementation Reading

Check:

- the chosen projection dimension $k$;
- whether $E$ and $F$ are shared across heads, layers, or key/value paths;
- whether projections are learned, fixed, convolutional, or otherwise constrained;
- whether causal masking is needed and how it interacts with sequence projection;
- maximum training length and test length;
- whether speedups include projection overhead;
- whether memory measurement includes activations, attention matrices, and optimizer state;
- whether comparisons use the same depth, width, objective, and token budget.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "Linformer proves all attention is low-rank." | It argues and tests a useful low-rank approximation regime; the assumption can fail. |
| "Linformer is the same as Performer." | Performer uses kernel random features; Linformer compresses key/value sequence length. |
| "Linear complexity means free long context." | $k$, projection overhead, kernels, and quality tradeoffs still matter. |
| "The output sequence is shortened." | The output remains length $n$; only the attended key/value memory is compressed. |
| "Low-rank attention is just LoRA." | Linformer compresses attention over sequence positions; LoRA adapts weight matrices with low-rank updates. |

## What to Remember

Linformer should be remembered as:

$$
\text{self-attention}
\rightarrow
\text{low-rank attention assumption}
\rightarrow
\text{sequence-axis projection of } K,V.
$$

It is useful because it gives an architecture-level way to ask:

$$
\text{How much pairwise token-token interaction rank does this task actually need?}
$$

That question remains relevant even when modern systems choose sparse attention, FlashAttention, recurrent memory, or state-space models instead.

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/longformer|Longformer]]
- [[papers/architectures/bigbird|BigBird]]
- [[papers/architectures/reformer|Reformer]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
- [[papers/architectures/lora|LoRA]]
