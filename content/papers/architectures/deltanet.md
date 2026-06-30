---
title: DeltaNet
aliases:
  - papers/deltanet
  - papers/parallelizing-linear-transformers-delta-rule
  - papers/delta-rule-linear-transformer
tags:
  - papers
  - architectures
  - transformer
  - attention
  - linear-attention
  - efficient-attention
  - recurrent
  - sequence-modeling
---

# DeltaNet

> DeltaNet matters because it turns a more expressive linear-attention update into something that can be trained efficiently over sequence length, closing part of the gap between recurrent linear models and softmax attention.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Parallelizing Linear Transformers with the Delta Rule over Sequence Length |
| Authors | Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim |
| Year | 2024 |
| Venue | NeurIPS 2024 |
| arXiv | [2406.06484](https://arxiv.org/abs/2406.06484) |
| OpenReview | [y8Rm4VNRPH](https://openreview.net/forum?id=y8Rm4VNRPH) |
| Implementation | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |
| Status | full note started |

## Question

Linear attention and state-space models aim for a useful compromise:

$$
\text{parallel training}
\quad+\quad
\text{linear-time recurrent inference}.
$$

For ordinary linear attention, a simplified recurrent update is:

$$
S_t
=
S_{t-1}
+
k_t^\top v_t,
\qquad
o_t=q_tS_t.
$$

This is efficient, but the additive update can be weak for tasks that require precise in-context retrieval or associative memory.

DeltaNet asks:

$$
\text{Can the recurrent memory update be more expressive while still trainable in parallel?}
$$

The answer is to use a delta-rule update and then derive a hardware-efficient parallel algorithm.

## Main Claim

The durable architecture claim is:

$$
\text{linear attention}
\xrightarrow{\text{delta rule}}
\text{more expressive matrix-state update}
\xrightarrow{\text{parallel algorithm}}
\text{scalable DeltaNet}.
$$

The paper reports that this makes DeltaNet competitive with, and in some reported settings stronger than, recent linear-time baselines such as [[papers/architectures/mamba|Mamba]] and [[papers/architectures/gated-linear-attention|GLA]].

For this wiki, the central lesson is:

$$
\text{recurrent sequence models need both a good update rule and a parallel training form}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Baseline family | linear Transformer / linear attention |
| Main update | delta-rule recurrent matrix-state update |
| Main algorithmic problem | parallelize training over sequence length |
| Mathematical tool | memory-efficient representation for products of Householder matrices |
| Scaling target | standard language-model training settings |
| Hybrid variants | DeltaNet plus sliding-window or global attention layers |
| Main comparison | softmax Transformer, GLA, Mamba, RetNet-style linear-time models |

The loss remains the causal language-model objective:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
$$

The contribution is in the sequence mixer and training algorithm.

## From Additive Memory to Delta Rule

In simple linear attention, the matrix memory is updated additively:

$$
S_t
=
S_{t-1}
+
k_t^\top v_t.
$$

That update stores new key-value information but does not explicitly correct an old prediction.

A delta-rule view instead updates memory using a prediction error. A simplified form is:

$$
\hat{v}_t
=
k_t S_{t-1},
$$

$$
e_t
=
v_t-\hat{v}_t,
$$

$$
S_t
=
S_{t-1}
+
\eta_t k_t^\top e_t.
$$

The exact paper implementation has additional parameterization and normalization details, but this is the useful reading:

$$
\text{write new association}
\quad\rightarrow\quad
\text{correct memory based on current error}.
$$

This makes DeltaNet closer to an online learning update over a recurrent memory state.

## Matrix-Valued RNN View

DeltaNet can be read as a matrix-valued recurrent neural network:

$$
S_t
=
F_t(S_{t-1}, k_t, v_t),
\qquad
o_t=q_tS_t.
$$

The hidden state is not a vector:

$$
S_t\in\mathbb{R}^{d_k\times d_v}.
$$

This is the same broad state shape family as [[papers/architectures/gated-linear-attention|GLA]], but the update rule is different.

| Model | State Update Intuition |
| --- | --- |
| linear attention | add key-value outer product |
| GLA | add key-value information with data-dependent forgetting |
| DeltaNet | update memory through a delta-rule correction |

That difference matters for retrieval. A recurrent state can be compact and efficient, but it must store the right associations.

## Parallelization Problem

A sequential recurrence is easy to define:

$$
S_1\rightarrow S_2\rightarrow \cdots \rightarrow S_T.
$$

It is hard to train efficiently on modern GPUs if every step must wait for the previous step.

The paper's algorithmic contribution is to parallelize DeltaNet over sequence length. The key point for paper reading is:

$$
\text{linear-time inference}
\not\Rightarrow
\text{hardware-efficient training}.
$$

A recurrent architecture needs a parallel or chunkwise training form. Otherwise, it can be theoretically efficient but practically slow.

## Householder and Compact Representation

The paper reparameterizes DeltaNet so that the recurrence can be handled through products of structured transformations, specifically using a memory-efficient representation for products of Householder matrices.

For the wiki, the exact linear algebra can be summarized as:

$$
\prod_{t=1}^{T}H_t
\quad
\text{should not be materialized naively}.
$$

Instead, a compact representation lets the algorithm avoid storing a full matrix-valued hidden state for every token during parallel training.

This is the same architecture-systems lesson as [[papers/architectures/flashattention|FlashAttention]] and [[papers/architectures/gated-linear-attention|GLA]]:

$$
\text{model family}
+
\text{algorithmic form}
+
\text{memory traffic}
=
\text{actual architecture}.
$$

## Hybrid DeltaNet

The paper also studies hybrid models that combine DeltaNet layers with softmax attention layers.

| Hybrid | Why It Matters |
| --- | --- |
| DeltaNet plus sliding-window attention | adds local precise token interactions |
| DeltaNet plus a few global attention layers | restores some full-context retrieval routes |

This is similar in spirit to [[papers/architectures/jamba|Jamba]], which combines Mamba, attention, and MoE. The broader lesson is:

$$
\text{efficient recurrent mixers}
\quad+\quad
\text{small amount of attention}
\quad\Rightarrow\quad
\text{strong hybrid backbone candidate}.
$$

## Relation to GLA

[[papers/architectures/gated-linear-attention|GLA]] improves linear attention with data-dependent gates and a hardware-aware training algorithm.

DeltaNet changes the update rule itself:

| Paper | Main Update |
| --- | --- |
| [GLA](/papers/architectures/gated-linear-attention) | gated linear-attention memory |
| [DeltaNet](/papers/architectures/deltanet) | delta-rule memory correction |

Both papers should be read together:

$$
\text{GLA}
\rightarrow
\text{how to gate and train linear attention efficiently},
$$

$$
\text{DeltaNet}
\rightarrow
\text{how to make the matrix-state update more expressive}.
$$

## Relation to Mamba and Mamba-2

Mamba-style models use selective state-space updates:

$$
h_t
=
A_t h_{t-1}
+
B_t x_t.
$$

DeltaNet uses a matrix-valued memory update over key-value-like features. Both are recurrent alternatives to dense softmax attention, but they compress history differently.

| Family | State |
| --- | --- |
| Mamba | selective SSM vector/state channels |
| Mamba-2 | SSD matrix-mixer view of SSM/attention duality |
| GLA | gated linear-attention matrix state |
| DeltaNet | delta-rule matrix memory |

This makes DeltaNet part of the same reading route as [[papers/architectures/mamba|Mamba]], [[papers/architectures/mamba-2|Mamba-2]], [[papers/architectures/retnet|RetNet]], and [[papers/architectures/hyena|Hyena]].

## Evidence to Read Carefully

The paper reports experiments at standard language-model scale, including a 1.3B model trained for 100B tokens, and compares against linear-time baselines and Transformer baselines.

Read the evidence by claim:

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| language-model perplexity | DeltaNet scales beyond toy tasks | universal superiority over softmax Transformers |
| zero-shot benchmarks | representation and generation utility | all downstream tasks or all data regimes |
| retrieval-focused tasks | better associative memory than simpler linear recurrences | perfect long-context retrieval |
| throughput | parallel algorithm is practical | speed independent of implementation and hardware |
| hybrid results | attention plus DeltaNet can be strong | pure recurrent models always suffice |

The main durable result is not a single leaderboard number. It is that a stronger recurrent memory update can be made parallel enough for large-scale language modeling.

## Practical Checks

- Identify whether a paper uses additive linear attention, gated linear attention, delta-rule updates, retention, or SSM.
- Track whether recurrent state is vector-valued or matrix-valued.
- Ask whether the training algorithm parallelizes over sequence length.
- Separate inference complexity from training wall-clock speed.
- Check retrieval-heavy tasks separately from average language-model perplexity.
- For hybrid models, count how many layers use full attention, sliding attention, or recurrent linear updates.
- Compare with matched tokenizer, data, parameter count, training tokens, and implementation.

## Where It Fits

DeltaNet is a useful next paper after GLA:

$$
\text{Linformer/Performer}
\rightarrow
\text{GLA}
\rightarrow
\text{DeltaNet}
\rightarrow
\text{hybrid recurrent-attention backbones}.
$$

It also gives a good template for reading future linear-attention papers: do not only ask whether the recurrence is linear in $T$; ask what memory update is used and whether that update has a practical parallel training algorithm.

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/linformer|Linformer]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
- [[papers/architectures/gated-linear-attention|Gated Linear Attention]]
- [[papers/architectures/retnet|RetNet]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/mamba-2|Mamba-2]]
- [[papers/architectures/jamba|Jamba]]
