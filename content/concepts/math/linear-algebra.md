---
title: Linear Algebra
tags:
  - math
  - linear-algebra
---

# Linear Algebra

Linear algebra is the language of vectors, matrices, projections, embeddings, and neural network layers. Most deep learning tensors are collections of linear algebra objects.

A vector is:

$$
x \in \mathbb{R}^{d}
$$

A matrix maps one vector space to another:

$$
y = Wx + b
$$

where $W \in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ and $b \in \mathbb{R}^{d_{\mathrm{out}}}$.

In neural networks, $x$ is rarely just one vector. It is usually a row inside a batch, token sequence, graph node table, image feature map, or coordinate matrix. Linear algebra becomes useful when the axes are explicit.

## Common Operations

Dot product:

$$
x^\top y = \sum_{i=1}^{d}x_i y_i
$$

Norm:

$$
\|x\|_2 = \sqrt{x^\top x}
$$

See [[concepts/math/vector-norm-similarity|Vector norm and similarity]] for dot product, cosine similarity, Euclidean distance, and normalized embeddings.

Matrix multiplication:

$$
C_{ij} = \sum_k A_{ik}B_{kj}
$$

Matrix multiplication is a structured way to combine many dot products at once. In ML this appears as feature projection, attention score computation, message aggregation, covariance estimation, and classifier logits.

## Shape Discipline

Most implementation errors are shape errors. A batch of vectors is often represented as:

$$
X
\in
\mathbb{R}^{B\times d}
$$

and a linear layer applies:

$$
Y
=
XW^\top + \mathbf{1}b^\top
$$

where $W\in\mathbb{R}^{d_{\mathrm{out}}\times d}$, $Y\in\mathbb{R}^{B\times d_{\mathrm{out}}}$, and $\mathbf{1}\in\mathbb{R}^{B}$ broadcasts the bias.

For sequence models:

$$
X
\in
\mathbb{R}^{B\times T\times d}
$$

where $B$ is batch size, $T$ is sequence length, and $d$ is feature dimension. The same vector operation may be applied independently over $B$ and $T$.

For graph or structure models:

$$
X
\in
\mathbb{R}^{B\times N\times d}
$$

where $N$ may be atoms, residues, graph nodes, or detected objects.

For coordinates:

$$
X_{\mathrm{coord}}
\in
\mathbb{R}^{B\times N\times 3}
$$

The last dimension is not just another feature dimension when geometry matters. Rotations and translations act on the coordinate axis, while node or residue features live in a different representation space.

## Mixing Axes

A matrix multiplication should be interpreted by the axis it mixes:

- Feature mixing: $y=xW$ changes channels/features at one position.
- Token mixing: attention uses a $T\times T$ matrix to mix sequence positions.
- Node mixing: graph methods use neighborhoods or adjacency-like matrices to mix nodes.
- Coordinate transforms: rotations mix the 3 coordinate axes while preserving distances.

This distinction prevents a common error: assuming every matrix multiplication has the same modeling meaning.

## Projection and Basis

A basis is a coordinate system for representing vectors. A projection maps a vector into a subspace.

If $U\in\mathbb{R}^{d\times k}$ has orthonormal columns, the projection onto the subspace spanned by $U$ is:

$$
P_U x
=
UU^\top x
$$

The residual is:

$$
r
=
x - P_U x
$$

This is the geometric idea behind PCA, linear probes, bottleneck representations, and low-dimensional diagnostics.

In deep models, a learned projection is often just:

$$
z = xW
$$

but the interpretation depends on the destination space: embedding space, query space, key space, value space, latent space, or task-logit space.

## Linear Maps and Subspaces

A matrix maps an input vector into an output subspace:

$$
\operatorname{col}(A)
=
\{Az: z\in\mathbb{R}^{n}\}
$$

The rank of a matrix is the dimension of the information it can transmit:

$$
\operatorname{rank}(A)
\le
\min(m,n)
$$

Low-rank layers, PCA, embedding compression, and representation collapse all use this idea.

If $W$ is factorized as:

$$
W
\approx
AB
$$

with $A\in\mathbb{R}^{d_{\mathrm{in}}\times r}$, $B\in\mathbb{R}^{r\times d_{\mathrm{out}}}$, and small $r$, then the transformation must pass through an $r$-dimensional bottleneck. This can reduce parameters and memory, but it also constrains the update or representation.

## Gram Matrix and Similarity Matrix

A Gram matrix stores pairwise inner products:

$$
G
=
XX^\top,
\quad
G_{ij}
=
x_i^\top x_j
$$

This appears as:

- attention logits before scaling and masking
- embedding similarity matrices
- kernel methods
- clustering and retrieval diagnostics
- protein, molecule, or document neighborhood analysis

When embeddings are normalized, the Gram matrix stores cosine similarities.

## Covariance View

For centered data matrix $X\in\mathbb{R}^{n\times d}$, the empirical covariance matrix is:

$$
\Sigma
=
\frac{1}{n-1}X^\top X
$$

Covariance connects linear algebra to statistics. Eigenvectors of $\Sigma$ give PCA directions, while eigenvalues describe variance along those directions.

Spectral structure:

- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]] describes special directions of square matrices.
- [[concepts/math/singular-value-decomposition|Singular value decomposition]] describes directions and scales for any matrix.

## Coordinate Matrices

For a molecule, protein backbone, ligand pose, or point cloud:

$$
X
\in
\mathbb{R}^{N\times 3}
$$

Each row is a point in 3D. A rigid transformation is:

$$
X'
=
XR^\top + \mathbf{1}t^\top
$$

where $R$ is a rotation matrix and $t$ is a translation vector.

Distances are preserved when $R$ is orthogonal:

$$
R^\top R = I
$$

This is the linear algebra behind [[concepts/math/geometry|Geometry]], [[concepts/math/symmetry-group|Symmetry group]], [[concepts/geometric-deep-learning/equivariance|Equivariance]], and structure-based evaluation.

## Why It Matters

- [[concepts/architectures/linear-layer|Linear layers]] are affine maps.
- [[concepts/architectures/attention|Attention]] uses matrix products between queries and keys.
- [[concepts/llm/embedding-retrieval|Embedding retrieval]] uses vector similarity.
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]] often finds lower-dimensional linear subspaces.
- [[concepts/geometric-deep-learning/index|Geometric deep learning]] uses coordinate matrices, rotations, frames, and equivariant maps.
- [[concepts/machine-learning/kernel-method|Kernel methods]] use inner products in explicit or implicit feature spaces.

## Checks

- What are the tensor shapes?
- Which dimensions are batch, sequence, feature, head, node, or coordinate axes?
- Is a transformation linear, affine, normalized, or nonlinear?
- Is similarity a dot product, cosine similarity, distance, or learned score?
- Is a matrix a feature projection, token mixing matrix, adjacency-like matrix, covariance matrix, or coordinate transform?
- Are tensors written in batch-first, sequence-first, or channel-first order?
- Is the operation preserving rank, reducing dimension, or mixing axes?
- Are coordinates, features, heads, and tokens being multiplied along the intended axes?
- Does a coordinate transform preserve distances, angles, orientation, or none of them?

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
