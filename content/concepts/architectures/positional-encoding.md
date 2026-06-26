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

## Position Contract

| Encoding | Encodes | Good For | Risk |
| --- | --- | --- | --- |
| absolute index | token position $t$ | fixed-length sequences | weak length extrapolation |
| relative bias | offset $i-j$ | local and long sequence patterns | offset bucket choices matter |
| rotary | relative phase in query/key space | long-context language models | extrapolation depends on scaling |
| segment/chain id | boundary or component identity | paired sequences, proteins, documents | missing boundaries cause leakage |
| graph distance | shortest path or edge distance | graphs and molecules | graph construction affects meaning |
| 3D distance/frame | spatial relation | structures and geometry | must respect rotation/translation |

The right encoding depends on the symmetry of the object. A set model, sequence model, graph model, and coordinate model should not use the same position story by default.

## Sequence Length Extrapolation

If a model is trained up to length $L_{\mathrm{train}}$ and evaluated at $L_{\mathrm{test}}>L_{\mathrm{train}}$, the paper should explain:

$$
p_t \quad \text{for} \quad t > L_{\mathrm{train}}
$$

Absolute learned embeddings may be undefined or extrapolated poorly. Relative or rotary methods often behave better, but still need empirical checks at the target context length.

## Structured Position

For protein, molecule, and graph inputs, position can mean several different things:

| Object | Position Type |
| --- | --- |
| protein sequence | residue index, chain id, MSA row/column |
| protein structure | residue index, 3D coordinate, local frame |
| molecule graph | atom index, graph distance, bond type, conformer coordinate |
| protein-ligand complex | receptor frame, ligand frame, pair distance |
| document retrieval | chunk order, source document, citation span |

Do not treat arbitrary input order as meaningful unless the representation contract says so.

## Checks

- Does the task need order, distance, geometry, or only set membership?
- Can the model extrapolate beyond the training context length?
- Are padding, chain breaks, and multiple sequences handled correctly?
- For protein or molecular structures, is sequence position enough, or is geometry also needed?
- Are absolute coordinates being used where an invariant or equivariant representation is required?
- Does the benchmark test longer contexts or different graph/structure sizes than training?

## Related

- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/modalities/representation-contract|Representation contract]]
