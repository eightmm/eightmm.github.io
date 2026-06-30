---
title: Longformer
aliases:
  - papers/longformer
  - papers/long-document-transformer
  - papers/sliding-window-attention
  - papers/longformer-encoder-decoder
tags:
  - papers
  - architectures
  - transformer
  - attention
  - long-context
  - sparse-attention
---

# Longformer

> Longformer replaces dense full attention with sliding-window local attention plus task-motivated global attention so Transformers can process long documents.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Longformer: The Long-Document Transformer |
| Authors | Iz Beltagy, Matthew E. Peters, Arman Cohan |
| Year | 2020 |
| Venue | arXiv preprint |
| arXiv | [2004.05150](https://arxiv.org/abs/2004.05150) |
| Code | [allenai/longformer](https://github.com/allenai/longformer) |
| Status | full note started |

## One-Line Takeaway

Longformer makes the attention pattern sparse:

$$
\text{dense attention}
\quad
O(T^2)
\qquad\rightarrow\qquad
\text{sliding window + global attention}
\quad
O(Tw + Tg),
$$

where $T$ is sequence length, $w$ is local window size, and $g$ is the number of global tokens.

## Question

Standard self-attention compares every token to every other token:

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
\mathrm{attention\ memory}=O(T^2).
$$

For long documents, this is the limiting factor:

$$
T=512
\quad\rightarrow\quad
T=4096
\quad
\text{makes dense attention much more expensive}.
$$

The paper asks:

$$
\text{Can a Transformer read long documents without full pairwise attention?}
$$

Longformer answers by making most attention local and reserving full-range attention for selected global tokens.

## Main Claim

The durable architecture claim is:

$$
\text{local window attention}
+ \text{global task tokens}
\rightarrow
\text{long document Transformer}.
$$

The method is a drop-in replacement for full self-attention:

$$
\operatorname{Attention}_{\text{dense}}
\quad\rightarrow\quad
\operatorname{Attention}_{\text{sparse}}.
$$

The rest of the Transformer block can remain familiar: residual stream, normalization, feed-forward network, and stacking.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | long token sequence, usually documents |
| Main operator | sparse self-attention |
| Local pattern | each token attends to a fixed-size sliding window |
| Global pattern | selected task tokens attend broadly and can be attended by all tokens |
| Complexity target | linear in sequence length for fixed window and global-token count |
| Main use | long-document classification, QA, language modeling, summarization with LED |
| Main risk | attention pattern must match the task's needed information flow |

Longformer is not a new modality. It is a Transformer attention pattern for long sequence inputs.

## Sliding-Window Attention

For token position $i$, define a local window radius $r$:

$$
\mathcal{N}_{\text{local}}(i)
=
\{j:\lvert i-j\rvert\leq r\}.
$$

The window size is:

$$
w=2r+1.
$$

Instead of attending to all keys:

$$
j\in\{1,\ldots,T\},
$$

Longformer attends to:

$$
j\in\mathcal{N}_{\text{local}}(i).
$$

The sparse attention output is:

$$
y_i
=
\sum_{j\in\mathcal{N}_{\text{local}}(i)}
\operatorname{softmax}_{j}
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
\right)v_j.
$$

For fixed $w$, local attention cost scales as:

$$
O(Twd_k).
$$

This is linear in $T$ if $w$ is treated as a constant.

## Global Attention

Local windows alone make long-range information flow slow. A token can only pass information across the sequence after many layers:

$$
\text{receptive field}
\approx
Lw
$$

for $L$ layers.

Longformer adds global tokens:

$$
\mathcal{G}\subseteq\{1,\ldots,T\}.
$$

For a global token $i\in\mathcal{G}$:

$$
\mathcal{N}(i)=\{1,\ldots,T\}.
$$

For ordinary token $i\notin\mathcal{G}$:

$$
\mathcal{N}(i)
=
\mathcal{N}_{\text{local}}(i)\cup\mathcal{G}.
$$

This creates a route for task-specific long-range aggregation:

$$
\text{local context}
\leftrightarrow
\text{global tokens}
\leftrightarrow
\text{distant local context}.
$$

## Mask View

Longformer can be read as a structured attention mask. Let:

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
\frac{q_i^\top k_j}{\sqrt{d_k}}
+M_{i,j}.
$$

This places Longformer inside the same general attention contract as [[concepts/architectures/attention|Attention]]:

$$
\text{attention}
=
\text{content score}
+\text{mask or bias}.
$$

## Relation to Dense Attention

| Attention Pattern | Each Token Reads | Complexity Sketch | Typical Use |
| --- | --- | --- | --- |
| dense | all tokens | $O(T^2)$ | short sequence Transformer |
| local window | nearby tokens | $O(Tw)$ | long sequence locality |
| dilated window | spaced nearby tokens | $O(Tw)$ | larger receptive field |
| global attention | selected full-range tokens | $O(Tg)$ | task aggregation |
| Longformer | local plus global | $O(Tw+Tg)$ | long documents |

Dense attention is more flexible, but it pays for all pairwise interactions. Longformer pays for a chosen communication graph.

## Relation to Other Architecture Notes

| Paper | Relation |
| --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | baseline dense Transformer attention |
| [Transformer-XL](/papers/architectures/transformer-xl) | long context through recurrence and segment memory |
| [BigBird](/papers/architectures/bigbird) | long context through local, global, and random sparse attention |
| [Reformer](/papers/architectures/reformer) | long sequence efficiency through LSH attention and reversible layers |
| [ALiBi](/papers/architectures/alibi) | length extrapolation through attention logit bias |
| [Performer](/papers/architectures/performer) | approximate softmax attention with linear complexity |
| [FlashAttention](/papers/architectures/flashattention) | exact dense attention with IO-aware kernel optimization |

Longformer is a sparse-pattern route, not an exact dense attention kernel route and not a positional extrapolation route.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| sparse attention scales to long documents | complexity and implementation experiments | long sequences can be processed without full $T^2$ attention | sparse pattern must match the task |
| local plus global works for NLP tasks | downstream long-document tasks | global tokens help task-level aggregation | global-token choice is task-dependent |
| pretraining matters | Longformer vs RoBERTa comparisons | long-context architecture benefits from pretraining | data and initialization matter |
| LED supports seq2seq long input | summarization experiments | sparse encoder attention can support long document generation | decoder and cross-attention still need careful cost accounting |

## Implementation Reading

Check:

- What is the local window size $w$?
- Are attention windows one-sided, two-sided, or dilated?
- Which tokens receive global attention?
- Does each task define global tokens consistently?
- Are padding and document boundaries masked correctly?
- Does the implementation materialize dense $T\times T$ masks anyway?
- Are results compared against truncation, chunking, and dense baselines at the same length?

The main failure mode is thinking the architecture "sees everything" because the document fits. In Longformer, each ordinary token sees a sparse neighborhood plus global routes.

## Limitations

- Sparse attention encodes a communication pattern; if the task needs arbitrary pairwise interactions, local windows may miss them.
- Global token selection is task-dependent and can become a hidden design choice.
- Linear scaling holds when window size and global token count are controlled.
- Long input support does not remove memory costs from embeddings, activations, FFNs, or decoder-side generation.
- Long-document benchmarks can reward input length rather than better reasoning if truncation baselines are weak.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "Longformer is just a bigger BERT." | Its key architecture change is the sparse local/global attention pattern. |
| "Linear attention means the same thing as Performer." | Longformer is sparse exact attention over selected positions; Performer approximates softmax attention. |
| "Every token attends globally." | Only selected global tokens attend broadly; ordinary tokens use local windows plus global connections. |
| "Longer input alone proves better architecture." | Compare against truncation, chunking, retrieval, and dense baselines when possible. |

## What to Remember

Longformer is the canonical sparse-attention paper for long documents:

$$
\text{full attention graph}
\rightarrow
\text{local window graph}
+ \text{task global nodes}.
$$

For this wiki, keep it as the anchor for windowed sparse attention, separate from ALiBi/RoPE position mechanisms, Performer-style attention approximation, and FlashAttention kernel optimization.

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/alibi|ALiBi]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
