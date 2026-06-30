---
title: BigBird
aliases:
  - papers/bigbird
  - papers/big-bird
  - papers/random-sparse-attention
  - papers/block-sparse-attention
tags:
  - papers
  - architectures
  - transformer
  - attention
  - long-context
  - sparse-attention
---

# BigBird

> BigBird replaces full self-attention with a sparse graph that combines local, global, and random links, giving long-sequence Transformers a linear-cost attention route.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Big Bird: Transformers for Longer Sequences |
| Authors | Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed |
| Year | 2020 |
| Venue | NeurIPS 2020 |
| arXiv | [2007.14062](https://arxiv.org/abs/2007.14062) |
| Proceedings | [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) |
| Code | [google-research/bigbird](https://github.com/google-research/bigbird) |
| Status | full note started |

## One-Line Takeaway

BigBird changes the attention graph:

$$
\text{full attention}
\quad
O(T^2)
\qquad\rightarrow\qquad
\text{local + global + random sparse attention}
\quad
O(T).
$$

The key object is the neighbor set for each query position:

$$
\mathcal{N}(i)
=
\mathcal{N}_{\text{local}}(i)
\cup
\mathcal{N}_{\text{global}}(i)
\cup
\mathcal{N}_{\text{random}}(i).
$$

## Question

Dense self-attention computes every pairwise token interaction:

$$
A
=
\frac{QK^\top}{\sqrt{d_k}}
\in
\mathbb{R}^{T\times T}.
$$

This gives:

$$
\mathrm{cost}=O(T^2d_k),
\qquad
\mathrm{memory}=O(T^2).
$$

For long documents, protein-like sequences, or genome-scale sequences, the quadratic term is the bottleneck. BigBird asks whether sparse attention can keep enough long-range connectivity to be useful.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | long sequence of tokens |
| Main operator | block sparse self-attention |
| Local route | each block attends to nearby blocks |
| Global route | selected tokens or blocks attend broadly |
| Random route | each block attends to selected random blocks |
| Complexity target | linear in sequence length when sparse degree is fixed |
| Main paper uses | long-document QA, summarization, classification, genomics examples |
| Main risk | sparse graph may not match the task's needed interactions |

BigBird is an attention-pattern paper. It changes the information-flow graph inside a Transformer while leaving the broad Transformer stack recognizable.

## Sparse Attention Graph

Let token positions be vertices:

$$
V=\{1,\ldots,T\}.
$$

Dense attention uses a complete directed graph:

$$
E_{\text{dense}}
=
\{(i,j): i,j\in V\}.
$$

BigBird uses:

$$
E_{\text{BigBird}}
=
E_{\text{local}}
\cup
E_{\text{global}}
\cup
E_{\text{random}}.
$$

For each query $i$, attention is computed only over its neighbor set:

$$
y_i
=
\sum_{j\in\mathcal{N}(i)}
\operatorname{softmax}_{j}
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right)v_j.
$$

Equivalently, use a sparse mask:

$$
M_{i,j}
=
\begin{cases}
0, & j\in\mathcal{N}(i),\\
-\infty, & \text{otherwise}.
\end{cases}
$$

Then:

$$
A_{i,j}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}+M_{i,j}.
$$

This keeps BigBird inside the same [[concepts/architectures/attention|Attention]] equation while changing which interactions exist.

## Three Sparse Routes

### Local Attention

Local attention handles nearby structure:

$$
\mathcal{N}_{\text{local}}(i)
=
\{j:\lvert i-j\rvert\le r\}.
$$

This is useful for phrase-level, sentence-level, motif-level, or local sequence patterns.

### Global Attention

Global tokens or blocks provide an aggregation path:

$$
\mathcal{G}\subseteq V.
$$

For a global token $g\in\mathcal{G}$:

$$
\mathcal{N}(g)=V.
$$

For ordinary tokens, global tokens can act as routing hubs:

$$
\mathcal{G}\subseteq\mathcal{N}(i).
$$

This resembles the task-token route in [[papers/architectures/longformer|Longformer]], but BigBird uses it as one part of a broader sparse graph.

### Random Attention

Random links create non-local shortcuts:

$$
\mathcal{N}_{\text{random}}(i)
\sim
\operatorname{Sample}(V, r_{\text{rand}}).
$$

Local windows alone pass information slowly across a long sequence:

$$
\text{receptive field after }L\text{ layers}
\approx
Lw.
$$

Random edges shorten communication paths:

$$
\text{local paths}
\quad\rightarrow\quad
\text{local paths plus non-local shortcuts}.
$$

## Block Sparse View

Split a sequence into blocks:

$$
X=[B_1,B_2,\ldots,B_m].
$$

Attention is defined between blocks:

$$
\mathcal{N}(B_i)
=
\mathcal{N}_{\text{local}}(B_i)
\cup
\mathcal{N}_{\text{global}}(B_i)
\cup
\mathcal{N}_{\text{random}}(B_i).
$$

If each block attends to a fixed number of other blocks, the number of attended pairs grows linearly with sequence length:

$$
|E_{\text{sparse}}|=O(T).
$$

This assumes block size and sparse degree are controlled. If the number of global or random links grows with $T$, the cost changes.

## Relation to Longformer

| Paper | Sparse Pattern | Main Difference |
| --- | --- | --- |
| [Longformer](/papers/architectures/longformer) | local sliding window plus task global tokens | emphasizes long-document practicality and task-specific global tokens |
| BigBird | local plus global plus random sparse blocks | adds random links and stronger theoretical expressivity claims |

Both are sparse-attention routes. Neither is the same as [[papers/architectures/performer|Performer]], which approximates softmax attention, or [[papers/architectures/flashattention|FlashAttention]], which computes exact dense attention more efficiently.

## Expressivity Claim

The paper's unusual contribution is that it does not only report an efficient sparse pattern. It argues that BigBird preserves important theoretical properties of full Transformer attention under its sparse construction.

The high-level claim is:

$$
\text{sparse attention with global and random routes}
\Rightarrow
\text{strong sequence-function expressivity}.
$$

This matters because sparse attention can otherwise look like a pure engineering compromise:

$$
\text{less compute}
\quad
\text{but maybe weaker model}.
$$

BigBird's argument is that a carefully designed sparse graph can reduce cost while retaining enough connectivity for broad sequence modeling.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| sparse attention reduces quadratic cost | complexity analysis and implementation | long sequences can be processed with fixed sparse degree | wall-clock gains depend on sparse kernel quality |
| local/global/random pattern works empirically | QA, summarization, classification experiments | sparse graph can support long-document tasks | improvements can depend on input length and baseline truncation |
| global tokens matter theoretically | theoretical analysis | global routes help preserve expressivity properties | theory assumptions are not the same as practical benchmark coverage |
| genomics can benefit from long context | sequence experiments | non-text long sequences are plausible targets | broad genomics stays out of this wiki unless tied to a concrete modeling problem |

## Implementation Reading

Check:

- block size and maximum sequence length;
- number of random blocks per query block;
- which tokens or blocks are global;
- whether random attention is fixed, seeded, or resampled;
- whether the implementation materializes dense masks before sparse computation;
- padding behavior at document or sequence boundaries;
- whether comparison baselines use truncation, chunking, retrieval, Longformer, Performer, or dense attention at shorter length.

The practical failure mode is treating "linear attention" as a single category. BigBird is sparse exact attention over selected positions, not kernelized softmax attention and not an IO-aware dense attention kernel.

## Relation to Other Architecture Notes

| Paper | Relation |
| --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | dense Transformer baseline |
| [Longformer](/papers/architectures/longformer) | closest local/global sparse-attention neighbor |
| [Reformer](/papers/architectures/reformer) | approximate similarity buckets through LSH attention plus reversible layers |
| [Transformer-XL](/papers/architectures/transformer-xl) | long context through segment recurrence rather than sparse attention |
| [Linformer](/papers/architectures/linformer) | long context through low-rank key/value sequence projection |
| [Performer](/papers/architectures/performer) | linear attention through kernel feature approximation |
| [FlashAttention](/papers/architectures/flashattention) | exact dense attention with IO-aware implementation |
| [ALiBi](/papers/architectures/alibi) | length extrapolation by attention bias rather than sparse graph design |

## Limitations

- Sparse graph design is a modeling assumption; missed edges can matter.
- Random links make implementation and reproducibility details more important.
- Linear asymptotic cost does not remove embedding, FFN, activation, or optimizer memory.
- Strong theoretical statements do not automatically prove better task performance.
- Long-context gains can be inflated if baselines are mostly truncation-limited.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "BigBird is just Longformer with a bigger window." | BigBird adds random sparse links and emphasizes theoretical properties. |
| "Linear attention always means the same method." | BigBird is sparse attention; Performer is kernelized attention; FlashAttention is exact dense attention. |
| "Random attention means unstructured noise." | The random links are a sparse graph design for non-local connectivity. |
| "The model fully reads all token pairs." | Only selected local, global, and random pairs are attended. |

## What to Remember

BigBird is the canonical paper for sparse attention as a graph-design problem:

$$
\text{long sequence Transformer}
=
\text{local structure}
+
\text{global hubs}
+
\text{random shortcuts}.
$$

Keep it next to [[papers/architectures/longformer|Longformer]] when reading long-context sparse attention, and separate it from approximate attention, positional extrapolation, and kernel-level acceleration.

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/longformer|Longformer]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
