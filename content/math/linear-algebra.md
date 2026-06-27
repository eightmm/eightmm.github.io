---
title: Linear Algebra
tags:
  - math
  - linear-algebra
---

# Linear Algebra

Linear algebra is the language of vectors, matrices, embeddings, projections, and learned representations.

$$
y = Wx + b
$$

This is the basic form behind [[concepts/architectures/linear-layer|linear layers]], many classifiers, and local transformations inside larger architectures.

## Core Notes

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/embedding|Embedding]]

## Shape Discipline

Most AI linear algebra is tensor linear algebra. The same operation can mean different things depending on which axis is being mixed.

Common shapes:

$$
X_{\mathrm{batch}}
\in
\mathbb{R}^{B\times d}
$$

$$
X_{\mathrm{sequence}}
\in
\mathbb{R}^{B\times T\times d}
$$

$$
X_{\mathrm{graph}}
\in
\mathbb{R}^{B\times N\times d}
$$

$$
X_{\mathrm{coords}}
\in
\mathbb{R}^{N\times 3}
$$

where $B$ is batch size, $T$ is sequence length, $N$ is number of nodes, atoms, or residues, and $d$ is feature dimension.

The key question is: which axis is being mixed?

- Feature mixing: linear layers and MLPs mix the $d$ dimension.
- Token mixing: attention mixes positions through an attention matrix.
- Node mixing: graph models mix neighborhoods through edges or adjacency.
- Coordinate transforms: rotations act on the 3D coordinate axis.

## Projection

A projection maps an object into a useful subspace. A linear projection is:

$$
z = xW
$$

where $W$ changes the representation basis or dimension. In AI, projection appears as:

- embedding projection
- query, key, and value projection in [[concepts/architectures/attention|Attention]]
- PCA projection for analysis
- low-dimensional bottlenecks
- task heads and linear probes

An orthogonal projection onto a subspace with basis matrix $U$ can be written as:

$$
P_U x
=
UU^\top x
$$

when the columns of $U$ are orthonormal.

## Rank and Low-Rank Structure

The rank of a matrix measures how many independent directions it can represent:

$$
\operatorname{rank}(W)
\le
\min(d_{\mathrm{in}},d_{\mathrm{out}})
$$

Low-rank structure appears in:

- PCA and SVD diagnostics
- embedding compression
- low-rank adapters such as LoRA-style updates
- representation collapse checks
- protein or molecule similarity matrices with redundant structure

A low-rank factorization replaces a large matrix with two smaller matrices:

$$
W
\approx
AB,
\quad
A\in\mathbb{R}^{d_{\mathrm{in}}\times r},
\quad
B\in\mathbb{R}^{r\times d_{\mathrm{out}}},
\quad
r \ll \min(d_{\mathrm{in}},d_{\mathrm{out}})
$$

This reduces parameters but also constrains what transformations can be represented.

## Attention as Linear Algebra

Self-attention is mostly matrix multiplication plus softmax:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
S
=
\frac{QK^\top}{\sqrt{d_k}}
$$

$$
Y
=
\operatorname{softmax}(S)V
$$

Here $S$ is a similarity matrix between elements. It is not just a formula detail; it defines which tokens, residues, atoms, or retrieved chunks can exchange information.

## Structure-Based Coordinate Matrices

Structure-based modeling often represents coordinates as:

$$
X
=
\begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_N^\top
\end{bmatrix}
\in
\mathbb{R}^{N\times 3}
$$

A rigid transform is:

$$
X'
=
XR^\top + \mathbf{1}t^\top
$$

where $R\in\mathbb{R}^{3\times 3}$ is a rotation matrix and $t\in\mathbb{R}^{3}$ is a translation vector.

This connects linear algebra to [[math/geometry-symmetry|Geometry and symmetry]], [[concepts/geometric-deep-learning/equivariance|Equivariance]], and structure evaluation.

## AI Connections

- Embedding similarity uses dot products, cosine similarity, and vector norms.
- Attention scores use matrix products before softmax.
- PCA, low-rank structure, and representation diagnostics use eigenvectors and SVD.
- Linear probes test whether a representation already contains task-relevant information.
- Coordinate models use matrices for positions, rotations, rigid transforms, and distance geometry.
- GPU bottlenecks often come from large matrix multiplications and memory layout.

## Checks

- What are the shapes of vectors and matrices?
- Is a transformation mixing features, tokens, nodes, or channels?
- Is a projection reducing dimension, changing basis, or creating Q/K/V features?
- Is similarity normalized or scale-sensitive?
- Is a low-rank approximation a model assumption or only an analysis tool?
- Does a coordinate transform preserve distances and angles?
- Is a matrix multiplication compute-bound, memory-bound, or communication-bound?

## Related

- [[math/index|Math]]
- [[ai/architectures|Architectures]]
- [[math/geometry-symmetry|Geometry and symmetry]]
- [[math/numerical-computing|Numerical computing]]
- [[concepts/learning/linear-probing|Linear probing]]
