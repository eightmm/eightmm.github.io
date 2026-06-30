---
title: Attention
tags:
  - architectures
  - attention
  - sequence-modeling
---

# Attention

Attention computes context-dependent interactions between elements. It is the core mixing mechanism in [[concepts/architectures/transformer|Transformers]] and appears in graph, multimodal, retrieval, and agent systems.

Scaled dot-product attention is:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Here $Q$ contains queries, $K$ contains keys, $V$ contains values, and $d_k$ is the key dimension. The softmax term defines which elements are mixed.

The score $QK^\top$ is a batched [[concepts/math/vector-norm-similarity|dot-product similarity]] between query and key vectors.

For an input sequence $X\in\mathbb{R}^{T\times d_{\mathrm{model}}}$, self-attention first projects tokens into queries, keys, and values:

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V
$$

The attention logits and weights are:

$$
A = \frac{QK^\top}{\sqrt{d_k}} + M
$$

$$
P = \operatorname{softmax}(A)
$$

$$
Y = PV
$$

Here $M$ is an optional mask. In causal attention, future positions receive $-\infty$ before softmax so each token only attends to previous tokens.

## Shape Contract

For batched multi-head attention:

$$
Q\in\mathbb{R}^{B\times H\times T_q\times d_k},
\qquad
K\in\mathbb{R}^{B\times H\times T_k\times d_k},
\qquad
V\in\mathbb{R}^{B\times H\times T_k\times d_v}
$$

The attention weights have shape:

$$
P
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)
\in
\mathbb{R}^{B\times H\times T_q\times T_k}
$$

and the output is:

$$
Y=PV
\in
\mathbb{R}^{B\times H\times T_q\times d_v}
$$

The softmax axis is the key axis $T_k$. This means each query distribution sums to one over candidate keys:

$$
\sum_{j=1}^{T_k} P_{b,h,i,j}=1
$$

Multi-head attention repeats this with separate projections:

$$
\operatorname{head}_h
= \operatorname{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})
$$

$$
\operatorname{MHA}(X)
= \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O
$$

## Mask Semantics

The mask $M$ changes the allowed information flow:

| Mask | Allows | Typical Use |
| --- | --- | --- |
| none | every query attends to every key | bidirectional encoder |
| causal | only current and previous positions | autoregressive decoder |
| padding | ignores padded keys | variable-length batches |
| local/window | attends only nearby positions | long sequence efficiency |
| block/sparse | attends selected blocks or patterns | scalable long context |
| graph/structure | attends only permitted neighbors or relations | graph and structure models |

Masking is part of the task contract. A model with future-token access is not solving the same autoregressive task as a causal model.

## Compute and Memory

Dense attention forms a $T_q\times T_k$ score matrix. For self-attention with $T_q=T_k=T$:

$$
\mathrm{memory}(P)=O(BHT^2)
$$

and the dominant attention score cost is:

$$
O(BHT^2d_k)
$$

This is why long-context models use sparse attention, chunking, recurrence, state-space models, retrieval, or KV caching.

## Attention Is Not Explanation

Attention weights show how values are mixed in a particular layer and head:

$$
y_i=\sum_{j}P_{ij}v_j
$$

They do not by themselves prove causal importance. Value vectors, later layers, residual paths, normalization, and MLP blocks can all change the final behavior.

## Key Ideas

- Queries ask what information is needed; keys decide what can be matched; values carry the mixed information.
- Self-attention mixes elements within one set, such as tokens in a sequence or nodes in a graph.
- Cross-attention mixes information across two sets, such as encoded inputs and generated outputs.
- Attention weights are useful debugging signals, but they are not always faithful explanations.
- Positional or structural encodings matter because plain attention is permutation-aware but not order-aware by itself.

## Practical Checks

- Check whether attention is full, causal, local, sparse, or cross-modal.
- Track tensor shapes: batch, heads, query length, key length, and head dimension.
- Watch memory cost when sequence length, graph size, or pair count grows.
- For papers, identify what is being attended over: tokens, residues, atoms, edges, retrieved chunks, or tools.
- Check which axis softmax normalizes over.
- Confirm whether masks are additive, multiplicative, causal, padding, graph, or learned.
- Separate attention weights from causal explanation.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/math/tensor-shape-notation|Tensor shape notation]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[agents/index|Agents]]
