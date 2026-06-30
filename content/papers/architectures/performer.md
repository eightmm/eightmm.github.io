---
title: Performer
aliases:
  - papers/performer
  - papers/rethinking-attention-with-performers
tags:
  - papers
  - architectures
  - transformer
  - attention
  - efficient-attention
---

# Performer

> The paper introduced FAVOR+ random features to approximate softmax attention with linear time and memory complexity.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Rethinking Attention with Performers |
| Authors | Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller |
| Year | 2020 preprint; ICLR 2021 |
| Venue | ICLR 2021 |
| arXiv | [2009.14794](https://arxiv.org/abs/2009.14794) |
| OpenReview | [Ua6zuk0WRH](https://openreview.net/forum?id=Ua6zuk0WRH) |
| Google Research | [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/) |
| Status | verified |

## Question

Standard Transformer attention builds an all-pairs attention matrix. For a sequence of length $n$, this creates quadratic time and memory pressure:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
$$

The attention matrix has shape:

$$
QK^\top\in\mathbb{R}^{n\times n}.
$$

The core question is:

$$
\text{Can we approximate full softmax attention without explicitly materializing } n^2 \text{ pairwise scores?}
$$

Performer answers this with a kernelized attention view and a random-feature approximation called FAVOR+.

## Main Claim

Performer estimates regular softmax attention with linear time and memory complexity in sequence length, without requiring sparsity or low-rank assumptions over the attention matrix.

The claim can be summarized as:

$$
\operatorname{softmax}(q^\top k)
\approx
\phi(q)^\top \phi(k)
$$

where $\phi(\cdot)$ is a positive random feature map.

Then attention can be rearranged:

$$
\operatorname{softmax}(QK^\top)V
\approx
\Phi(Q)\left(\Phi(K)^\top V\right),
$$

so the large $n\times n$ matrix does not need to be formed.

The durable architecture idea is:

$$
\text{attention as a kernel}
\rightarrow
\text{random feature approximation}
\rightarrow
\text{linear sequence scaling}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | token sequence represented by $Q,K,V$ |
| Baseline operation | softmax self-attention |
| Replacement | FAVOR+ approximate attention |
| Main efficiency goal | avoid explicit $n\times n$ attention matrix |
| Complexity target | linear in sequence length for fixed feature dimension |
| Key assumption | attention kernel can be approximated by positive random features |
| Main risk | approximation quality and implementation constants matter |

Performer is a Transformer-compatible attention replacement. It is not a new language-model objective, tokenizer, or data recipe.

## Standard Attention Cost

Let:

$$
Q,K,V\in\mathbb{R}^{n\times d}.
$$

Softmax attention is:

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right),
$$

$$
Y=AV.
$$

The score matrix costs:

$$
O(n^2d)
$$

to compute and:

$$
O(n^2)
$$

to store.

For long sequences, the dominant problem is not the projection layers. It is the pairwise token-token matrix.

## Kernel Attention View

If the attention kernel can be written as:

$$
\kappa(q,k)
=
\phi(q)^\top\phi(k),
$$

then unnormalized attention can be written:

$$
\tilde y_i
=
\sum_{j=1}^{n}
\kappa(q_i,k_j)v_j
=
\sum_{j=1}^{n}
\phi(q_i)^\top\phi(k_j)v_j.
$$

By associativity:

$$
\tilde y_i
=
\phi(q_i)^\top
\left(
\sum_{j=1}^{n}
\phi(k_j)v_j^\top
\right).
$$

The sum over keys and values can be computed once:

$$
S
=
\sum_{j=1}^{n}
\phi(k_j)v_j^\top
=
\Phi(K)^\top V.
$$

Then:

$$
\tilde Y
=
\Phi(Q)S.
$$

This changes the computation order:

$$
(\Phi(Q)\Phi(K)^\top)V
\quad\rightarrow\quad
\Phi(Q)(\Phi(K)^\top V).
$$

The first expression forms an $n\times n$ matrix. The second does not.

## Normalization

Softmax attention is row-normalized. Kernelized attention needs a corresponding denominator:

$$
y_i
=
\frac{
\sum_{j=1}^{n}\phi(q_i)^\top\phi(k_j)v_j
}{
\sum_{j=1}^{n}\phi(q_i)^\top\phi(k_j)
}.
$$

Using associativity:

$$
y_i
=
\frac{
\phi(q_i)^\top
\left(
\sum_j\phi(k_j)v_j^\top
\right)
}{
\phi(q_i)^\top
\left(
\sum_j\phi(k_j)
\right)
}.
$$

This is the key linear-attention structure. The numerator and denominator use prefix-like or full-sequence summaries rather than an explicit pairwise matrix.

## FAVOR+

FAVOR+ stands for Fast Attention Via positive Orthogonal Random features. The method approximates the softmax kernel with positive random features and uses orthogonal random features to reduce variance.

The ideal identity is:

$$
\mathbb{E}_\omega
\left[
\phi_\omega(q)^\top\phi_\omega(k)
\right]
=
\exp(q^\top k).
$$

In practice, a finite number $m$ of random features is used:

$$
\phi(q)\in\mathbb{R}^{m}.
$$

The approximation improves as $m$ grows, but compute and memory also grow:

$$
O(ndm)
\quad\text{or}\quad
O(nm d_v)
$$

depending on the implementation details and value dimension.

The paper's important contribution is not merely "use random features." It is a positive, low-variance random feature construction suitable for attention normalization.

## Complexity Comparison

For sequence length $n$, head dimension $d$, value dimension $d_v$, and feature count $m$:

| Attention Type | Time Shape | Memory Shape | Main Bottleneck |
| --- | --- | --- | --- |
| full softmax attention | $O(n^2d+n^2d_v)$ | $O(n^2)$ | pairwise attention matrix |
| sparse/local attention | depends on window/sparsity | depends on pattern | chosen connectivity |
| Performer attention | roughly linear in $n$ for fixed $m$ | no explicit $n^2$ matrix | random-feature dimension and kernel quality |

The architecture tradeoff is:

$$
\text{exact pairwise softmax}
\rightarrow
\text{approximate kernelized attention}.
$$

This is useful when $n$ is large enough that quadratic attention is the main constraint.

## Relation to Transformer

Performer keeps the Transformer skeleton:

$$
x'
=
x+\operatorname{Attention}(\operatorname{Norm}(x)),
$$

$$
y
=
x'+\operatorname{FFN}(\operatorname{Norm}(x')).
$$

The replacement is inside the attention operator:

$$
\operatorname{SoftmaxAttention}
\rightarrow
\operatorname{FAVORAttention}.
$$

This makes it an attention-mechanism paper rather than a full-stack language-model paper.

## Relation to Sparse Attention

Sparse attention reduces cost by restricting which token pairs interact:

$$
A_{ij}=0
\quad
\text{for many pairs }(i,j).
$$

Performer does not choose a sparse pattern. It approximates dense attention through a kernel map:

$$
A
\approx
\Phi(Q)\Phi(K)^\top.
$$

| Approach | How Cost Is Reduced | Risk |
| --- | --- | --- |
| local attention | attend only nearby tokens | misses distant relations unless stacked or shifted |
| block sparse attention | attend selected blocks | pattern design matters |
| [Linformer](/papers/architectures/linformer) | compress keys and values along sequence length | rank assumption may fail |
| Performer | random-feature kernel approximation | approximation variance and kernel mismatch |

The reading point is that "efficient attention" is not one method family. Each method changes the connectivity or approximation contract differently.

## Relation to Mamba and SSMs

[[papers/architectures/mamba|Mamba]] and other state-space sequence models reduce sequence cost by replacing attention with recurrent/scan-like state updates. Performer keeps an attention-style kernelized aggregation.

| Axis | Performer | Mamba / SSM |
| --- | --- | --- |
| starting point | Transformer attention | state-space recurrence |
| sequence scaling | linear for fixed feature count | linear scan |
| memory mechanism | kernelized all-token aggregation | learned state transition and selective updates |
| exact softmax attention | approximate | not attention |
| core question | can dense attention be approximated cheaply? | can sequence modeling avoid attention? |

This makes Performer a useful bridge between vanilla Transformer attention and non-attention sequence architectures.

## Evidence to Read

Performer was evaluated across tasks where long inputs or dense attention cost matter, including image/pixel-style sequences, language modeling, and protein sequence modeling.

Read the evidence around these questions:

| Evidence | What It Supports |
| --- | --- |
| approximation theory | FAVOR+ estimates attention kernels with guarantees |
| long-sequence experiments | linear attention is useful when quadratic attention is costly |
| comparisons to sparse/dense efficient attention | the method is competitive with other efficiency routes |
| kernel comparisons | different attention kernels can be studied at larger scale |
| protein sequence tasks | long biological sequences can benefit from efficient attention |

The strongest evidence is not only final task score. It is the combination of:

$$
\text{scaling behavior}
+
\text{approximation quality}
+
\text{downstream task performance}.
$$

## What to Watch in the Ablations

Efficient attention papers are easy to misread. Check:

| Question | Why It Matters |
| --- | --- |
| Is sequence length large enough? | quadratic attention may not be the bottleneck at small $n$ |
| Is the implementation optimized? | asymptotic complexity can lose to constants |
| Is memory or wall-clock speed reported? | FLOPs alone may hide kernel overhead |
| How many random features are used? | approximation quality and compute both depend on $m$ |
| Is the baseline exact softmax or another efficient method? | the claim changes with the comparator |
| Does the task require sharp token-token alignment? | approximation may blur important interactions |

## Failure Modes and Caveats

- Linear asymptotic complexity does not guarantee faster wall-clock time for every sequence length or hardware.
- Random-feature dimension $m$ introduces an accuracy/compute tradeoff.
- Approximate attention can behave differently from exact softmax attention in tasks requiring precise alignment.
- Kernel choice matters; Performer is a framework for kernelizable attention, not a universal replacement.
- Efficient attention claims should be read alongside implementation details, memory format, and batching.

## Why This Matters for Architecture Reading

Performer makes attention a kernel approximation problem:

$$
\text{attention weights}
\rightarrow
\text{kernel values}
\rightarrow
\text{feature maps}
\rightarrow
\text{linear aggregation}.
$$

That lens is useful even when a later model does not use Performer directly. It separates three questions:

1. What pairwise similarity should attention use?
2. Can that similarity be approximated without building $n^2$ scores?
3. Does the approximation preserve the task-relevant interaction?

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/math/linear-algebra|Linear algebra]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/roformer|RoFormer]]
- [[papers/architectures/mamba|Mamba]]

## One-Line Memory

Performer is the linear-attention paper that treats softmax attention as a kernel and uses FAVOR+ random features to avoid forming the quadratic attention matrix.
