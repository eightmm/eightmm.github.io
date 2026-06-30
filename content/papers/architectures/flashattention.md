---
title: FlashAttention
aliases:
  - papers/flashattention
  - papers/flash-attention
  - papers/fast-and-memory-efficient-exact-attention
tags:
  - papers
  - architectures
  - attention
  - systems
---

# FlashAttention

> The paper introduced an IO-aware exact attention algorithm that avoids materializing the full attention matrix in high-bandwidth memory, making Transformer attention faster and more memory-efficient without changing model semantics.

## Metadata

| Field | Value |
| --- | --- |
| Paper | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| Authors | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re |
| Year | 2022 |
| Venue | NeurIPS 2022 |
| arXiv | [2205.14135](https://arxiv.org/abs/2205.14135) |
| OpenReview | [H4DqfPSibmx](https://openreview.net/forum?id=H4DqfPSibmx) |
| Code | [Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention) |
| Status | full note started |

## One-Line Takeaway

FlashAttention keeps exact softmax attention but changes the computation schedule: tile $Q,K,V$, stream blocks through fast on-chip memory, maintain online softmax statistics, and avoid writing the $N\times N$ attention matrix to HBM.

## Question

Standard scaled dot-product attention is:

$$
O
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V,
$$

where:

$$
Q,K,V\in\mathbb{R}^{N\times d}.
$$

The attention matrix is:

$$
S
=
\frac{QK^\top}{\sqrt{d}}
\in
\mathbb{R}^{N\times N}.
$$

The architecture/systems question is:

$$
\text{Can exact attention be computed without storing the full }N\times N\text{ matrix in slow GPU memory?}
$$

FlashAttention answers yes by making attention IO-aware.

## Main Claim

Transformer attention is often bottlenecked by memory traffic, not only by FLOPs. An exact attention algorithm that minimizes reads and writes between GPU HBM and SRAM can reduce memory use and wall-clock time while preserving the same mathematical attention output.

The durable pattern is:

$$
\text{same attention function}
+
\text{different memory schedule}
\rightarrow
\text{faster and longer-context Transformer}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | query, key, value matrices $Q,K,V$ |
| Output | same exact attention output $O$ as standard softmax attention in real arithmetic |
| Main trick | tiling, online softmax, recomputation for backward pass |
| Does it approximate attention? | no, the main algorithm computes exact attention |
| Main resource target | HBM reads/writes and activation memory |
| Architecture role | drop-in attention primitive for Transformers |
| Domain | long-context language models, vision Transformers, sequence models, attention-heavy architectures |

This is not a new attention score:

$$
\mathrm{FlashAttention}(Q,K,V)
\equiv
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
$$

It is a new way to compute that expression.

## Why Standard Attention Wastes Memory

Naive attention often materializes intermediate matrices:

$$
S=QK^\top,
\qquad
P=\mathrm{softmax}(S),
\qquad
O=PV.
$$

The matrix $P$ has $N^2$ entries:

$$
\mathrm{memory}(P)
=
O(N^2).
$$

Even if matrix multiplication is fast, moving $S$ and $P$ between memory levels can become the bottleneck.

| Memory level | Role | Problem |
| --- | --- | --- |
| SRAM / shared memory | fast, small, on-chip | cannot hold full attention matrix |
| HBM | slower, large GPU memory | expensive to repeatedly read/write |
| registers | fastest, very small | only hold partial reductions |

FlashAttention optimizes the schedule around this hierarchy.

## Tiled Attention

Split $Q,K,V$ into blocks:

$$
Q
=
\begin{bmatrix}
Q_1\\
Q_2\\
\vdots
\end{bmatrix},
\qquad
K
=
\begin{bmatrix}
K_1\\
K_2\\
\vdots
\end{bmatrix},
\qquad
V
=
\begin{bmatrix}
V_1\\
V_2\\
\vdots
\end{bmatrix}.
$$

For a query block $Q_i$ and key/value block $(K_j,V_j)$, compute local scores:

$$
S_{ij}
=
\frac{Q_iK_j^\top}{\sqrt{d}}.
$$

Instead of writing all $S_{ij}$ and $P_{ij}$ to HBM, FlashAttention keeps block-level quantities in faster memory and updates the output incrementally.

The high-level loop is:

$$
\text{load block}
\rightarrow
\text{compute local scores}
\rightarrow
\text{update online softmax}
\rightarrow
\text{accumulate output}
\rightarrow
\text{discard local block}.
$$

## Online Softmax

Softmax seems to require all scores at once:

$$
\mathrm{softmax}(s)_j
=
\frac{\exp(s_j)}
{\sum_k \exp(s_k)}.
$$

FlashAttention uses online normalization so blocks can be processed without storing the full row.

For one query row, maintain:

$$
m
=
\max_j s_j,
\qquad
\ell
=
\sum_j \exp(s_j-m),
$$

and the weighted value accumulator:

$$
o
=
\sum_j \exp(s_j-m)v_j.
$$

When a new block arrives with block maximum $m_b$, update:

$$
m'
=
\max(m,m_b),
$$

$$
\ell'
=
e^{m-m'}\ell
+
e^{m_b-m'}\ell_b,
$$

$$
o'
=
e^{m-m'}o
+
e^{m_b-m'}o_b.
$$

The final output is:

$$
O
=
\frac{o}{\ell}.
$$

This preserves exact softmax semantics while streaming blocks.

## Backward Pass and Recomputation

Training needs gradients through attention. A naive implementation stores large intermediate matrices for backward:

$$
S,\quad P,\quad O.
$$

FlashAttention avoids storing $S$ and $P$ by recomputing block-level attention pieces during the backward pass. This trades extra compute for much lower memory traffic.

The key tradeoff is:

$$
\text{save }O(N^2)\text{ activations}
\quad\text{by recomputing tiled blocks}.
$$

On modern accelerators, this can be faster because memory movement is often more expensive than the additional arithmetic.

## Exact vs Approximate Attention

Many efficient-attention papers reduce cost by changing the attention function:

| Route | Example | Tradeoff |
| --- | --- | --- |
| low-rank approximation | approximate $QK^\top$ | may change model quality |
| kernelized attention | approximate softmax kernel | may change numerical behavior |
| sparse attention | restrict visible tokens | changes receptive field |
| FlashAttention | exact attention, new schedule | same function, hardware-aware implementation |

This distinction matters. FlashAttention is not a new inductive bias like local attention or sparse attention. It is a computational primitive that makes the standard Transformer more practical.

## Relation to Other Architecture Notes

| Paper | Similarity | Difference |
| --- | --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | defines scaled dot-product attention as the Transformer core | FlashAttention changes implementation, not the mathematical layer |
| [Performer](/papers/architectures/performer) | targets efficient long-sequence attention | Performer approximates softmax attention with random features |
| [Transformer-XL](/papers/architectures/transformer-xl) | helps long-context sequence modeling | Transformer-XL changes recurrence/memory; FlashAttention changes kernel scheduling |
| [S4](/papers/architectures/s4) | long-sequence efficiency motivation | S4 changes sequence layer family; FlashAttention keeps attention |
| [Mamba](/papers/architectures/mamba) | responds to attention scaling limits | Mamba replaces attention with selective state-space layers |
| [LLaMA](/papers/architectures/llama) | modern decoder-only Transformer recipe | LLaMA-style models often rely on efficient attention kernels in practice |

## Evidence to Read

| Evidence | What it supports | What it does not prove |
| --- | --- | --- |
| IO complexity analysis | HBM access matters and can be reduced | every hardware/kernel implementation is equally efficient |
| wall-clock training speedups | exact attention can be faster in real systems | speedup is constant across all sequence lengths and GPUs |
| memory savings | longer contexts or larger batches become feasible | model quality improves automatically |
| long-range benchmarks | efficient exact attention helps enable longer-context experiments | attention is always better than SSMs, recurrence, or sparse alternatives |

## Why This Matters

FlashAttention is a key architecture paper because it changes how to read Transformer scaling. The bottleneck is not only:

$$
\text{parameter count}
\quad\text{or}\quad
\text{FLOPs}.
$$

It is also:

$$
\text{memory movement}
\quad\text{and}\quad
\text{kernel schedule}.
$$

For long-context LLMs, the difference between "the layer is mathematically possible" and "the model can actually train or serve" often depends on kernels like FlashAttention.

## Limitations

FlashAttention does not remove the $O(N^2d)$ arithmetic cost of dense exact attention. It reduces memory traffic and activation memory, but dense attention still scales quadratically in sequence length.

Speedups depend on hardware, sequence length, head dimension, batch shape, precision, masking pattern, kernel implementation, and framework integration. A paper using FlashAttention should still report the exact setup.

FlashAttention is not an alignment, reasoning, retrieval, or agent method. It makes attention cheaper; it does not solve whether the model has the right data, objective, evaluation, or tool-use behavior.

## Common Misreadings

### "FlashAttention is approximate attention."

No. The main FlashAttention algorithm computes exact softmax attention, but schedules it differently.

### "FlashAttention makes attention linear."

No. Dense exact attention still has quadratic pairwise interactions. FlashAttention reduces memory traffic and memory footprint.

### "FlashAttention is only a systems trick."

It is a systems-aware architecture primitive. It changes what Transformer configurations are practical, especially for long-context training and inference.

## What to Remember

When reading a Transformer paper that claims long context or efficient attention, ask:

- Is attention exact, approximate, sparse, recurrent, or replaced?
- Does the method reduce FLOPs, memory traffic, activation memory, or all three?
- Is speed measured as kernel throughput, training throughput, or end-to-end latency?
- Does the result depend on sequence length, head dimension, batch shape, GPU generation, or precision?
- Are causal masks, attention bias, dropout, and backward pass included?
- Is the memory bottleneck HBM traffic, KV cache, activations, or parameters?

The compact mental model is:

$$
\text{FlashAttention}
=
\text{exact attention}
+
\text{tiling}
+
\text{online softmax}
+
\text{IO-aware GPU scheduling}.
$$

## Links

- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/llama|LLaMA]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
