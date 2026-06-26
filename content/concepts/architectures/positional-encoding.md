---
title: Positional Encoding
tags:
  - architectures
  - transformer
  - sequence-modeling
---

# Positional Encoding

Positional encoding injects order or location into a model that otherwise treats inputs as a set. It is essential for Transformers because plain self-attention is permutation-equivariant.

A common absolute sinusoidal encoding is:

$$
p_{t,2i} = \sin\left(\frac{t}{10000^{2i/d}}\right)
$$

$$
p_{t,2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

The input token state is then:

$$
x_t = e_t + p_t
$$

where $e_t$ is a token embedding and $p_t$ is the positional encoding.

Relative position bias modifies the attention logits directly:

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + b_{i-j}
\right)V
$$

Rotary position embeddings apply a position-dependent rotation to query and key vectors:

$$
\operatorname{score}_{ij}
=
\frac{(R_i q_i)^\top (R_j k_j)}{\sqrt{d_k}}
$$

This makes attention depend on relative offsets through the rotated inner product.

## Variants

- Absolute learned positions attach one vector to each index.
- Relative position bias modifies attention scores based on distance.
- Rotary position embeddings rotate query/key features as a function of position.
- Structural encodings can use graph distance, residue separation, 3D distance, or chain identity.

## Checks

- Does the task need order, distance, geometry, or only set membership?
- Can the model extrapolate beyond the training context length?
- Are padding, chain breaks, and multiple sequences handled correctly?
- For protein or molecular structures, is sequence position enough, or is geometry also needed?

## Related

- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/math/symmetry-group|Symmetry group]]
