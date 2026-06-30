---
title: Reformer
aliases:
  - papers/reformer
  - papers/the-efficient-transformer
  - papers/lsh-attention
  - papers/reversible-transformer
tags:
  - papers
  - architectures
  - transformer
  - attention
  - efficient-transformer
  - long-context
---

# Reformer

> Reformer makes Transformers practical on long sequences by combining locality-sensitive hashing attention with reversible residual layers.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Reformer: The Efficient Transformer |
| Authors | Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya |
| Year | 2020 |
| Venue | ICLR 2020 |
| arXiv | [2001.04451](https://arxiv.org/abs/2001.04451) |
| OpenReview | [rkgNKkHtvB](https://openreview.net/forum?id=rkgNKkHtvB) |
| Status | full note started |

## One-Line Takeaway

Reformer attacks two different Transformer bottlenecks:

$$
\text{attention cost}
\quad
O(L^2)
\rightarrow
O(L\log L)
$$

through locality-sensitive hashing attention, and:

$$
\text{activation memory across }N\text{ layers}
\quad
O(N)
\rightarrow
O(1)
$$

through reversible residual layers.

## Question

Standard Transformer self-attention forms:

$$
A=\frac{QK^\top}{\sqrt{d_k}}
\in
\mathbb{R}^{L\times L}.
$$

This gives:

$$
\mathrm{attention\ time}=O(L^2d_k),
\qquad
\mathrm{attention\ memory}=O(L^2).
$$

Training also stores intermediate activations for backpropagation:

$$
\mathrm{activation\ memory}
\propto
N\cdot L\cdot d_{\mathrm{model}},
$$

where $N$ is the number of layers. Reformer asks whether a Transformer can keep similar modeling behavior while reducing both long-sequence attention cost and depth-related activation memory.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | long token sequence |
| Main attention operator | LSH attention |
| Key memory operator | reversible residual layer |
| Additional memory trick | chunked feed-forward computation |
| Main efficiency target | lower attention cost and lower activation storage |
| Main use case | long sequence modeling where dense attention and deep activation storage are limiting |
| Main risk | approximate hashing changes the attention neighborhood and can affect training dynamics |

Reformer is not only an attention paper. It is an efficient Transformer architecture package:

$$
\text{Reformer}
=
\text{LSH attention}
+
\text{reversible residual layers}
+
\text{chunked FFN computation}.
$$

## LSH Attention

Dense attention compares every query to every key. Reformer instead tries to compare each query mostly to keys likely to be similar.

Let a hash function map vectors to buckets:

$$
h(x)\in\{1,\ldots,B\}.
$$

The intended property is:

$$
\operatorname{sim}(q_i,k_j)\text{ high}
\quad\Rightarrow\quad
h(q_i)=h(k_j)
\text{ with higher probability}.
$$

Then each query attends within a smaller candidate set:

$$
\mathcal{N}_{\mathrm{LSH}}(i)
=
\{j:h(q_i)=h(k_j)\}.
$$

The attention output becomes:

$$
y_i
=
\sum_{j\in\mathcal{N}_{\mathrm{LSH}}(i)}
\operatorname{softmax}_{j}
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right)v_j.
$$

This is approximate attention. It does not compute all pairwise scores.

## Shared Query-Key Space

Reformer uses the same vector as query and key for LSH attention:

$$
Q=K.
$$

This matters because hashing should group vectors that are similar under the same comparison space. If queries and keys came from unrelated projections, hash buckets would be harder to interpret as approximate nearest-neighbor groups.

The basic intuition is:

$$
\text{same representation space}
\rightarrow
\text{hash buckets approximate attention neighborhoods}.
$$

## Sorting and Chunking

After hashing, tokens are sorted by hash bucket. Attention is then computed in chunks containing nearby sorted positions.

At a high level:

$$
X
\xrightarrow{\text{hash}}
\text{bucket ids}
\xrightarrow{\text{sort}}
\text{bucket-contiguous sequence}
\xrightarrow{\text{chunked attention}}
Y.
$$

This makes the computation more structured than arbitrary sparse attention. The model avoids computing a full $L\times L$ attention matrix, but it still uses dot-product attention inside each selected bucket or chunk.

## Multiple Hash Rounds

One hash round can miss useful neighbors. Reformer therefore can use multiple hash rounds:

$$
\mathcal{N}(i)
=
\bigcup_{r=1}^{R}
\mathcal{N}^{(r)}_{\mathrm{LSH}}(i).
$$

More rounds increase the chance that similar tokens share a bucket:

$$
R\uparrow
\quad\Rightarrow\quad
\text{better recall of similar keys}
\quad\text{but}\quad
\text{more compute}.
$$

This is the central quality-efficiency trade-off in LSH attention.

## Reversible Residual Layers

Standard residual blocks store activations:

$$
y=x+F(x).
$$

Backpropagation usually needs saved intermediate activations for every layer. Reversible layers split the hidden state:

$$
x=(x_1,x_2).
$$

A reversible block computes:

$$
y_1=x_1+F(x_2),
$$

$$
y_2=x_2+G(y_1).
$$

The input can be reconstructed from the output:

$$
x_2=y_2-G(y_1),
$$

$$
x_1=y_1-F(x_2).
$$

This means training can recompute activations during backward pass instead of storing every layer's activations:

$$
\text{store all layer activations}
\quad\rightarrow\quad
\text{store final activations and reconstruct}.
$$

The trade-off is extra recomputation.

## Chunked Feed-Forward Computation

Transformer feed-forward layers often expand dimension:

$$
d_{\mathrm{ff}}\gg d_{\mathrm{model}}.
$$

The FFN is token-wise:

$$
\operatorname{FFN}(x_i)
=
W_2\sigma(W_1x_i+b_1)+b_2.
$$

Because tokens can be processed independently in the FFN, Reformer can chunk this computation:

$$
X=[X^{(1)},X^{(2)},\ldots,X^{(C)}],
$$

and evaluate:

$$
\operatorname{FFN}(X)
=
[
\operatorname{FFN}(X^{(1)}),
\ldots,
\operatorname{FFN}(X^{(C)})
].
$$

This is an implementation-level memory saving. It should not change the numerical function if done exactly.

## Relation to Other Efficient Attention Notes

| Paper | Route | What Changes |
| --- | --- | --- |
| [Longformer](/papers/architectures/longformer) | sparse local/global attention | hand-designed local and task-global pattern |
| [BigBird](/papers/architectures/bigbird) | sparse local/global/random attention | graph connectivity through random sparse links |
| Reformer | LSH attention plus reversibility | approximate nearest-neighbor attention and activation reconstruction |
| [Linformer](/papers/architectures/linformer) | low-rank sequence projection | compresses keys and values before attention |
| [Performer](/papers/architectures/performer) | kernel feature approximation | approximates softmax attention algebraically |
| [FlashAttention](/papers/architectures/flashattention) | IO-aware exact attention | keeps dense attention exact but avoids materializing full matrix in HBM |

Reformer is closest to approximate nearest-neighbor attention. It is not simply a sparse mask, and it is not an exact dense attention kernel.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| LSH attention reduces attention complexity | complexity analysis and runtime experiments | long sequences become more practical than dense attention | hash quality and implementation details matter |
| reversible layers reduce activation memory | architecture construction | depth-related activation storage can be reduced | recomputation cost is paid during backward pass |
| chunked FFN reduces memory | implementation equivalence | FFN intermediate memory can be controlled | mostly a memory engineering trick, not a new model function |
| performance stays close to Transformer | empirical tasks and ablations | efficiency changes can preserve useful modeling behavior | task coverage is narrower than modern LLM practice |

## Implementation Reading

Check:

- number of hash rounds $R$;
- bucket size and chunk size;
- whether causal masking is preserved after sorting;
- whether queries and keys are shared;
- whether attention is allowed to attend to duplicate or same-token positions;
- memory saved by reversibility versus extra recomputation time;
- whether benchmark gains come from longer sequence length rather than a better inductive bias.

The most important implementation detail is that sorting/hash bucketing must not break the task mask. Approximate neighbor selection is useful only if the resulting information flow still matches the problem.

## Limitations

- LSH attention is approximate; important token pairs may be missed.
- Hashing and sorting add implementation complexity.
- Reversible layers trade memory for recomputation.
- The architecture was influential, but many later long-context systems prefer sparse patterns, kernel tricks, FlashAttention-style exact kernels, or state-space models.
- Efficiency claims depend heavily on sequence length, hardware, batching, and implementation quality.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "Reformer is just another sparse attention mask." | LSH attention selects candidates by hash buckets, not only by a fixed position pattern. |
| "Reversible layers make the model free to train." | They reduce activation storage but require recomputation. |
| "LSH attention is exact attention." | It approximates the relevant attention neighborhood. |
| "All efficient Transformers solve the same problem." | Reformer targets approximate attention and activation memory; FlashAttention targets IO for exact dense attention. |

## What to Remember

Reformer is the canonical efficient Transformer paper for:

$$
\text{nearest-neighbor-like attention by hashing}
+
\text{activation memory reduction by reversibility}.
$$

Keep it as a distinct route from [[papers/architectures/longformer|Longformer]], [[papers/architectures/bigbird|BigBird]], [[papers/architectures/performer|Performer]], [[papers/architectures/flashattention|FlashAttention]], and [[papers/architectures/s4|S4]].

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/longformer|Longformer]]
- [[papers/architectures/bigbird|BigBird]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
