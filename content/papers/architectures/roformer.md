---
title: RoFormer
aliases:
  - papers/roformer
  - papers/architectures/rope
  - papers/rope
  - papers/rotary-position-embedding
  - papers/enhanced-transformer-with-rotary-position-embedding
tags:
  - papers
  - architectures
  - transformer
  - positional-encoding
---

# RoFormer

> The paper introduced Rotary Position Embedding, a way to encode position by rotating query and key vectors so attention scores depend naturally on relative position.

## Metadata

| Field | Value |
| --- | --- |
| Paper | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| Authors | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu |
| Year | 2021 preprint; 2023 journal version |
| Venue | arXiv preprint; Neurocomputing journal article |
| arXiv | [2104.09864](https://arxiv.org/abs/2104.09864) |
| DOI | [10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063) |
| Code | [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer) |
| Status | full note started |

## One-Line Takeaway

RoPE applies a position-dependent rotation to queries and keys:

$$
\tilde{q}_m = R_m q_m,
\qquad
\tilde{k}_n = R_n k_n,
$$

so their dot product becomes a function of both token content and relative offset:

$$
\tilde{q}_m^\top \tilde{k}_n
=
q_m^\top R_m^\top R_n k_n
=
q_m^\top R_{n-m} k_n.
$$

## Question

Transformers need positional information because self-attention over a sequence is otherwise permutation equivariant:

$$
\operatorname{Attn}(PX)=P\operatorname{Attn}(X)
$$

for a permutation matrix $P$ when no position signal is injected.

The architecture question is:

$$
\text{How should position enter attention?}
$$

Common options include:

| Method | Where Position Enters |
| --- | --- |
| absolute sinusoidal | added to token embeddings |
| learned absolute | added to token embeddings |
| relative bias | added to attention logits |
| relative key/value | modifies attention keys/values |
| RoPE | rotates query/key vectors before dot product |

RoFormer asks whether position can be encoded in a way that keeps the dot-product attention interface while making relative position appear naturally in the score.

## Main Claim

Rotary Position Embedding encodes absolute position through rotation matrices, while the attention score depends on relative position.

For token position $m$:

$$
q_m = x_m W_Q,
\qquad
k_m = x_m W_K.
$$

Apply rotation:

$$
\tilde{q}_m = R_m q_m,
\qquad
\tilde{k}_n = R_n k_n.
$$

Then:

$$
\tilde{q}_m^\top \tilde{k}_n
=
q_m^\top R_m^\top R_n k_n.
$$

Because rotations compose by offset:

$$
R_m^\top R_n = R_{n-m},
$$

the attention logit can depend on the relative position $n-m$.

The durable claim:

$$
\text{absolute-position rotation}
\Rightarrow
\text{relative-position attention score}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | ordered token sequence |
| Target block | self-attention query/key projection |
| Position operation | apply position-dependent rotations to $Q$ and $K$ |
| Value operation | values are usually not rotated |
| Attention score | dot product of rotated queries and keys |
| Symmetry change | breaks permutation equivariance by injecting order |
| Relative behavior | score depends on offset through rotation composition |
| Main use | language models and long-context Transformers |
| Later impact | widely adopted in LLaMA-style decoder-only models |

RoPE is not a separate model family. It is an architecture component for attention.

## Rotation In Two Dimensions

For a 2D feature pair, a rotation by angle $\theta$ is:

$$
R(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}.
$$

For position $m$, RoPE uses:

$$
R_m = R(m\theta).
$$

Given:

$$
q=
\begin{bmatrix}
q_1 \\
q_2
\end{bmatrix},
$$

the rotated query is:

$$
R(m\theta)q
=
\begin{bmatrix}
q_1\cos(m\theta)-q_2\sin(m\theta) \\
q_1\sin(m\theta)+q_2\cos(m\theta)
\end{bmatrix}.
$$

This is applied independently across feature pairs.

## High-Dimensional RoPE

For head dimension $d$, RoPE groups dimensions into pairs:

$$
(1,2),(3,4),\ldots,(d-1,d).
$$

Each pair uses its own frequency:

$$
\theta_i = 10000^{-2(i-1)/d},
\qquad
i=1,\ldots,d/2.
$$

The full rotation matrix is block diagonal:

$$
R_m
=
\operatorname{diag}
\left(
R(m\theta_1),
R(m\theta_2),
\ldots,
R(m\theta_{d/2})
\right).
$$

So RoPE can be implemented without materializing the full matrix. In vector form, it interleaves cosine and sine terms:

$$
\operatorname{RoPE}(x,m)
=
x\odot \cos(m\Theta)
+
\operatorname{rotate\_half}(x)\odot \sin(m\Theta).
$$

Here $\operatorname{rotate\_half}$ swaps each feature pair with a sign change:

$$
(x_1,x_2)\mapsto(-x_2,x_1).
$$

## Attention With RoPE

Standard scaled dot-product attention is:

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

With RoPE:

$$
\tilde{Q}_m = R_m Q_m,
\qquad
\tilde{K}_n = R_n K_n,
$$

and:

$$
\operatorname{Attn}_{m}
=
\sum_n
\operatorname{softmax}_n
\left(
\frac{
\tilde{Q}_m \tilde{K}_n^\top
}{
\sqrt{d_k}
}
+
\operatorname{mask}_{m,n}
\right)
V_n.
$$

Values are not rotated:

$$
V_n = X_n W_V.
$$

This matters when reading implementations. RoPE is a query/key positional operation, not a value transform.

## Relative Position Property

Because:

$$
R_m^\top R_n = R_{n-m},
$$

the score between position $m$ and $n$ becomes:

$$
\tilde{q}_m^\top \tilde{k}_n
=
q_m^\top R_{n-m} k_n.
$$

So the score includes relative offset without adding a separate learned bias table:

$$
\text{relative distance}
\rightarrow
\text{phase difference}.
$$

This is why RoPE fits causal language models. A token can compare its query with keys at different offsets through the same dot-product mechanism.

## Relation To Absolute And Relative Position

| Position Scheme | Mechanism | Strength | Risk |
| --- | --- | --- | --- |
| learned absolute | add $p_m$ to token embedding | simple within trained length | weak extrapolation beyond table |
| sinusoidal absolute | add deterministic $p_m$ | no learned table limit | position enters before all layers |
| relative bias | add $b_{m-n}$ to attention logit | explicit relative distance | separate bias mechanism |
| Transformer-XL relative attention | decomposed content and relative terms | memory-friendly relative positions | more complex attention formula |
| RoPE | rotate $Q,K$ by position | relative effect inside dot product | extrapolation still depends on frequency/scaling |

RoPE's practical appeal is that it preserves the standard attention API:

$$
Q,K,V
\rightarrow
\operatorname{softmax}(QK^\top)V
$$

after replacing $Q,K$ by rotated versions.

## Long-Context Reading

RoPE is often discussed as if it automatically solves long context. That is too strong.

The right claim is narrower:

$$
\text{RoPE gives a position rule that can be evaluated at unseen positions}.
$$

Whether a model works at longer context depends on:

| Factor | Why |
| --- | --- |
| training context length | model may not learn to use long offsets |
| frequency base | high/low frequencies control phase behavior |
| RoPE scaling method | later systems modify angles for extension |
| attention pattern | full, local, sparse, sliding, or recurrent context |
| data distribution | long-range dependencies must appear in training/eval |
| retrieval or memory | context may come from external systems, not RoPE alone |

Modern long-context models often use RoPE variants, but their context gains are not caused by RoPE alone.

## Relation To LLaMA-Style Models

[[papers/architectures/llama|LLaMA]] uses rotary positional embeddings as part of a modern decoder-only recipe:

$$
\text{RMSNorm}
+
\text{RoPE}
+
\text{SwiGLU}
+
\text{causal attention}
\Rightarrow
\text{efficient decoder-only foundation model}.
$$

RoFormer is the canonical note for the RoPE component. LLaMA is the canonical note for the full foundation-model recipe.

When reading a LLaMA-like paper, ask:

| Question | Why |
| --- | --- |
| Is RoPE applied to all heads or a subset? | implementation details differ |
| What base/frequency schedule is used? | affects context behavior |
| Is any RoPE scaling used? | context extension changes angles |
| Are $Q$ and $K$ both rotated? | rotating only one side changes the property |
| Are values rotated? | usually no |
| Is position reset across segments, documents, or packed sequences? | packed training can create boundary artifacts |

## Relation To Transformer-XL

[[papers/architectures/transformer-xl|Transformer-XL]] also targets long-context sequence modeling, but with a different mechanism:

| Axis | Transformer-XL | RoPE / RoFormer |
| --- | --- | --- |
| core idea | segment-level recurrence and memory | rotary query/key position encoding |
| position type | relative attention decomposition | rotation composition gives relative offset |
| memory | cached previous segment hidden states | none by itself |
| main impact | recurrent long-context Transformer | default positional method in many decoder LMs |

They can be conceptually combined: a memory mechanism still needs a positional scheme that handles offsets correctly.

## Evidence Pattern

The paper evaluates RoFormer on long text classification benchmarks and compares against positional encoding alternatives.

For architecture reading, the useful evidence categories are:

| Evidence | What It Supports |
| --- | --- |
| comparison to position baselines | RoPE is a competitive position-injection method |
| theoretical dot-product property | relative position emerges inside attention |
| long text classification | position handling matters for longer sequences |
| compatibility with linear attention | rotation can be applied before attention kernelization |

The strongest long-term contribution is the formulation, not the exact benchmark table. RoPE became important because it was simple, implementation-friendly, and worked well in later large decoder-only models.

## Why It Belongs In Architecture Papers

RoPE is small but architecture-critical. It changes the attention block's contract:

$$
(Q,K,V)
\rightarrow
(\operatorname{RoPE}(Q),\operatorname{RoPE}(K),V).
$$

This affects:

- sequence order;
- relative offsets;
- long-context extrapolation;
- KV cache behavior;
- implementation compatibility with efficient attention kernels;
- model-family recipes such as LLaMA.

It belongs in architecture papers because many modern architecture papers assume RoPE without explaining it.

## Practical Implementation Checks

| Check | Expected |
| --- | --- |
| dimension pairing | adjacent or split-half convention must match implementation |
| query/key only | values usually remain unrotated |
| frequency tensor | same positions and dtype/device as attention tensors |
| causal mask | RoPE does not replace masking |
| KV cache | cached keys should already be rotated at their original positions |
| packed sequences | position ids should reset or continue intentionally |
| long-context scaling | scaling rule should be named and evaluated |

If a model has strange degradation at long context, inspect position ids and RoPE scaling before blaming attention alone.

## Limitations

- RoPE is a positional encoding method, not a memory mechanism.
- Long-context extrapolation is not guaranteed just because rotations can be evaluated for larger $m$.
- Frequency choices and scaling rules matter.
- It does not solve attention's quadratic cost.
- It assumes a meaningful 1D sequence order.
- For graph, set, or 3D coordinate inputs, order may be artificial unless the representation contract defines it.

The concise limitation:

$$
\text{RoPE gives relative-aware attention scores}
\neq
\text{unlimited reliable context}.
$$

## What To Remember

- RoPE rotates queries and keys by position.
- The dot product of rotated vectors depends on relative position through $R_m^\top R_n=R_{n-m}$.
- Values are usually not rotated.
- RoPE preserves the standard dot-product attention interface.
- It became a default component in LLaMA-style decoder-only models.
- Long-context claims still need separate evidence about training length, scaling, data, and evaluation.

## Links

- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/modalities/sequence|Sequence]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/llama|LLaMA]]
