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

Additive attention bias is another common way to inject structure:

$$
A_{b,h,i,j}
=
\frac{q_{b,h,i}^{\top}k_{b,h,j}}{\sqrt{d_k}}
+M_{i,j}
+B_{h,i,j}.
$$

The bias $B$ can encode relative position, graph relation, distance bucket, or a fixed linear distance prior. [[papers/architectures/alibi|ALiBi]] is the canonical linear-bias paper note in this wiki.

Some architectures do more than add a bias. [[papers/architectures/deberta|DeBERTa]] decomposes attention scores into content-content and content-position terms, which is useful when position should not be collapsed into the same embedding vector as token identity.

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

## Key-Value Head Sharing

Modern decoder-only models often keep many query heads but use fewer key/value heads to reduce KV-cache cost during autoregressive decoding. The canonical paper note here is [[papers/architectures/gqa|GQA]].

Let $H_q$ be query heads and $H_{kv}$ be key/value heads:

| Variant | Head Contract | Main Effect |
| --- | --- | --- |
| MHA | $H_{kv}=H_q$ | each query head has its own key/value head |
| MQA | $H_{kv}=1$ | all query heads share one key/value head |
| GQA | $1 < H_{kv} < H_q$ | groups of query heads share key/value heads |

The attention formula is still dot-product attention, but the tensor shape changes:

$$
\operatorname{head}_h
=
\operatorname{Attention}
\left(
Q_h,\,
K_{g(h)},\,
V_{g(h)}
\right).
$$

This is an architecture decision because it changes the KV cache:

$$
\text{KV cache size}
\propto
L\cdot T\cdot H_{kv}\cdot d_h.
$$

## Mask Semantics

The mask $M$ changes the allowed information flow:

| Mask | Allows | Typical Use |
| --- | --- | --- |
| none | every query attends to every key | bidirectional encoder |
| causal | only current and previous positions | autoregressive decoder |
| padding | ignores padded keys | variable-length batches |
| local/window | attends only nearby positions | long sequence efficiency; see [Longformer](/papers/architectures/longformer) |
| block/sparse | attends selected blocks or patterns | scalable long context; see [BigBird](/papers/architectures/bigbird) |
| hash bucket | attends to tokens sharing approximate similarity buckets | efficient approximate attention; see [Reformer](/papers/architectures/reformer) |
| low-rank projection | compresses keys and values along the sequence axis | linear attention through compression; see [Linformer](/papers/architectures/linformer) |
| disentangled content-position | scores content and relative position with separate terms | position-aware encoder attention; see [DeBERTa](/papers/architectures/deberta) |
| graph/structure | attends only permitted neighbors or relations | graph and structure models |

Masking is part of the task contract. A model with future-token access is not solving the same autoregressive task as a causal model.

Local attention can also be used as one part of a hybrid recurrent backbone. [[papers/architectures/griffin|Griffin]] mixes gated linear recurrence with local attention:

$$
\text{local attention}
\quad+\quad
\text{fixed-size recurrent state}.
$$

This gives nearby tokens explicit softmax interaction while recurrent state carries compressed context.

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

Sparse attention changes which entries of the $T\times T$ attention graph are computed. For a window size $w$, a local attention pattern has the rough cost:

$$
O(BHTwd_k),
$$

which is linear in $T$ if $w$ is fixed. [[papers/architectures/longformer|Longformer]] is the canonical local-plus-global sparse attention paper note, while [[papers/architectures/bigbird|BigBird]] is the canonical local-plus-global-plus-random sparse attention paper note.

Hash-based attention chooses candidates by approximate similarity rather than fixed position:

$$
\mathcal{N}(i)=\{j:h(q_i)=h(k_j)\}.
$$

[[papers/architectures/reformer|Reformer]] is the canonical LSH attention note here.

Low-rank attention compresses the sequence axis before attention:

$$
\tilde K=EK,\qquad \tilde V=FV,
\qquad
E,F\in\mathbb{R}^{k\times T}.
$$

Then the score matrix changes from $T\times T$ to $T\times k$:

$$
\operatorname{Attention}_{\text{low-rank}}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{Q\tilde K^\top}{\sqrt{d_k}}
\right)
\tilde V.
$$

[[papers/architectures/linformer|Linformer]] is the canonical low-rank attention paper note here.

Kernel linear attention avoids the $T\times T$ matrix by rewriting attention through feature maps:

$$
\kappa(q,k)
=
\phi(q)^\top\phi(k),
$$

$$
Y
\approx
\Phi(Q)(\Phi(K)^\top V).
$$

[[papers/architectures/performer|Performer]] is the canonical random-feature approximation note. [[papers/architectures/gated-linear-attention|Gated Linear Attention]] is the canonical hardware-aware gated linear attention note; it reads linear attention as a parallel training form and a recurrent inference form:

$$
S_t
=
g_t\odot S_{t-1}
+
k_t^\top v_t,
\qquad
o_t=q_tS_t.
$$

[[papers/architectures/deltanet|DeltaNet]] is the canonical delta-rule linear Transformer note. It changes the recurrent memory update from simple additive writing toward error-correcting memory updates:

$$
S_t
=
S_{t-1}
+
\eta_t k_t^\top (v_t-k_tS_{t-1}).
$$

## Gated and Moving-Average Attention

Some attention variants keep content-dependent mixing but add a local sequential bias before or inside the attention block. [[papers/architectures/mega|Mega]] is the canonical paper note here.

An exponential moving average gives each token a decayed local memory:

$$
z_t
=
\alpha x_t
+
(1-\alpha)z_{t-1}.
$$

Mega combines this moving-average path with gated attention:

$$
\tilde{X}=\operatorname{EMA}(X),
\qquad
Y=\operatorname{GatedAttention}(\tilde{X}).
$$

This differs from a pure attention bias. A bias changes attention scores, while EMA adds a stateful local sequence operator.

## Attention and State-Space Duality

Some modern sequence papers compare attention and state-space models through the matrix they apply over sequence positions:

$$
Y = MX.
$$

In attention, $M$ is usually a content-dependent token-token matrix. In SSMs, $M$ can arise from expanding a recurrent state update across the whole prefix. [[papers/architectures/mamba-2|Mamba-2]] is the canonical note here for structured state space duality, which relates SSMs and attention-like mixers through structured semiseparable matrices.

This does not mean every Transformer is literally the same as every SSM. It means the right comparison often starts with:

$$
\text{what structure does the sequence mixing matrix have?}
$$

## Vision Attention Papers

| Paper | Why It Matters |
| --- | --- |
| [Vision Transformer](/papers/architectures/vision-transformer) | applies Transformer self-attention to image patches |
| [Swin Transformer](/papers/architectures/swin-transformer) | uses shifted local windows and hierarchy for vision attention |
| [CoAtNet](/papers/architectures/coatnet) | stages convolution and relative attention to balance bias and capacity |

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
- Attention biases are part of the architecture contract: a causal mask, relative position bias, ALiBi bias, graph mask, or structure mask changes which interactions are easy or possible.

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
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[papers/architectures/linformer|Linformer]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/gated-linear-attention|Gated Linear Attention]]
- [[papers/architectures/deltanet|DeltaNet]]
- [[papers/architectures/griffin|Griffin]]
- [[papers/architectures/mega|Mega]]
- [[papers/architectures/mamba-2|Mamba-2]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[agents/index|Agents]]
