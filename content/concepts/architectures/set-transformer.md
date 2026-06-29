---
title: Set Transformer
tags:
  - architectures
  - set-modeling
  - attention
---

# Set Transformer

Set Transformers apply attention to unordered sets. They extend [[concepts/architectures/deep-sets|Deep Sets]] by allowing pairwise or higher-order interactions between elements while preserving permutation symmetry.

For set elements $X=\{x_i\}_{i=1}^{n}$, self-attention over elements can be written as:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
Y=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Because the same projections and attention rule are applied to every element, the output is permutation equivariant. A final pooling or learned seed attention step can produce permutation-invariant set outputs.

For a permutation matrix $P$, a set model should satisfy one of two contracts.

Permutation-equivariant element output:

$$
f(PX)=P f(X)
$$

Permutation-invariant set output:

$$
g(PX)=g(X)
$$

Set Transformers usually build equivariant element interactions first, then use pooling or seed attention for invariant readout.

## Key Ideas

- Deep Sets pool independent element embeddings; Set Transformers let elements interact before readout.
- Attention cost grows with the number of set elements, so inducing points or latent bottlenecks are often used.
- They are useful for candidate sets, point sets, retrieved contexts, molecular conformer sets, and multi-instance inputs.
- The distinction from sequence Transformers is that element order should not carry meaning unless explicit positional features are added.

## Readout Patterns

| Output type | Contract | Typical readout |
| --- | --- | --- |
| set-level class or scalar | invariant | pooling, attention pooling, seed vectors |
| one output per element | equivariant | element-wise head after attention |
| top-k or ranking | equivariant scores plus sorting | per-element scoring with masking |
| variable-size structured output | task-specific | decoder with explicit output contract |

## Cost And Inducing Points

Full self-attention over $n$ set elements costs:

$$
O(n^2d_k)
$$

Inducing-point variants introduce $m$ learned or computed inducing tokens with $m \ll n$:

$$
H = \operatorname{Attn}(I, X),
\qquad
Y = \operatorname{Attn}(X, H)
$$

This changes the interaction cost toward $O(nmd_k)$, at the price of an information bottleneck through the inducing tokens.

## Failure Modes

| Failure | What to check |
| --- | --- |
| accidental order dependence | index, rank, or positional feature leaks an arbitrary order |
| bad masking | padded elements participate in attention or pooling |
| weak set summary | pooling hides rare but important elements |
| scaling issue | full pairwise attention becomes too expensive for large sets |

## Practical Checks

- Decide whether the target should be permutation invariant or equivariant.
- Check whether positional, rank, or index features accidentally introduce order dependence.
- Watch set size scaling and masking for variable-size sets.
- Specify whether readout is pooling, seed attention, per-element scoring, or decoder-based.
- Compare against [[concepts/architectures/graph-construction|graph construction]] when relationships are sparse or typed.

## Related

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/perceiver|Perceiver]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
