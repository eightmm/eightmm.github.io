---
title: Tensor Shape Notation
tags:
  - math
  - notation
  - tensors
---

# Tensor Shape Notation

Tensor shape notation records what each axis means before interpreting a formula. Many AI paper equations look simple because they hide batch, token, head, node, coordinate, or candidate axes.

A useful rule is:

$$
\text{shape}
=
\text{semantic axes}
+
\text{operation axes}
+
\text{broadcast axes}
$$

## Common Axes

| Symbol | Meaning | Example |
| --- | --- | --- |
| $B$ | batch size | independent examples |
| $T$ | sequence length | tokens, residues, time steps |
| $N$ | set, graph, atom, or node count | atoms, residues, graph nodes |
| $E$ | edge count | molecular bonds, contact edges |
| $C$ | channel count | image channels or feature maps |
| $D$ | feature dimension | embedding width |
| $H$ | attention heads | multi-head attention |
| $d_h$ | head dimension | per-head query/key/value width |
| $K$ | candidates or classes | retrieval items, labels, conformers |
| $R$ | coordinate dimension | usually $3$ for 3D coordinates |

## Batch and Feature Convention

A batch of feature vectors is often:

$$
X \in \mathbb{R}^{B \times D}
$$

A batch of token sequences is:

$$
X \in \mathbb{R}^{B \times T \times D}
$$

A coordinate set is:

$$
X \in \mathbb{R}^{B \times N \times 3}
$$

Do not treat the last axis as always "features." In coordinate models, the coordinate axis transforms under rotation; feature channels do not.

## Operation Axes

| Operation | Typical Shape | Mixed Axis |
| --- | --- | --- |
| Linear layer | $X\in\mathbb{R}^{B\times T\times D}$, $W\in\mathbb{R}^{D\times D'}$ | feature axis |
| Self-attention | $Q,K,V\in\mathbb{R}^{B\times H\times T\times d_h}$ | token axis through $QK^\top$ |
| Convolution | $X\in\mathbb{R}^{B\times C\times H_x\times W_x}$ | local spatial axes |
| GNN message passing | node features $H\in\mathbb{R}^{N\times D}$, edges $E$ | graph neighborhood |
| Pooling | $X\in\mathbb{R}^{B\times N\times D}\rightarrow z\in\mathbb{R}^{B\times D}$ | node/set/token axis |
| Pairwise distance | $X\in\mathbb{R}^{B\times N\times 3}$ | node pair and coordinate axes |

## Attention Shape

For multi-head self-attention:

$$
Q,K,V \in \mathbb{R}^{B\times H\times T\times d_h}
$$

The attention logits are:

$$
A_{b,h,i,j}
=
\frac{Q_{b,h,i,:}\cdot K_{b,h,j,:}}{\sqrt{d_h}}
$$

so:

$$
A \in \mathbb{R}^{B\times H\times T\times T}
$$

Softmax is taken over the key axis $j$, not over the batch, head, or feature axis:

$$
P_{b,h,i,j}
=
\frac{\exp(A_{b,h,i,j})}
{\sum_{j'=1}^{T}\exp(A_{b,h,i,j'})}
$$

## Graph and Molecular Shapes

Molecular and protein models often combine graph, coordinate, and feature axes:

| Object | Shape |
| --- | --- |
| Atom features | $H\in\mathbb{R}^{N_{\mathrm{atom}}\times D}$ |
| Bond edges | $E\in\{1,\ldots,N\}^{2\times M}$ |
| Atom coordinates | $X\in\mathbb{R}^{N_{\mathrm{atom}}\times 3}$ |
| Residue features | $R\in\mathbb{R}^{N_{\mathrm{res}}\times D}$ |
| Protein-ligand pair features | $P\in\mathbb{R}^{N_{\mathrm{prot}}\times N_{\mathrm{lig}}\times D}$ |

For batched graphs, shape is often ragged. A single tensor shape may hide a `batch_index`, padding mask, or packed edge list.

## Broadcasting

Broadcasting should be explicit in formulas that mix learned parameters and batched data. For layer normalization:

$$
Y_{b,t,d}
=
\gamma_d
\frac{X_{b,t,d}-\mu_{b,t}}{\sqrt{\sigma^2_{b,t}+\epsilon}}
+
\beta_d
$$

$\gamma,\beta\in\mathbb{R}^{D}$ are broadcast across $B$ and $T$.

## Checks

- What does each axis mean?
- Which axis is reduced by sum, mean, softmax, pooling, or normalization?
- Which axis is mixed by a matrix multiplication?
- Which axes are broadcast?
- Is the implementation batch-first, sequence-first, or channel-first?
- Are graph or molecular examples padded, packed, or ragged?
- Does a coordinate axis transform under rotation or translation?
- Does a reported loss average over batch, token, atom, pair, candidate, or graph units?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
