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

Spectral structure:

- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]] describes special directions of square matrices.
- [[concepts/math/singular-value-decomposition|Singular value decomposition]] describes directions and scales for any matrix.

## Why It Matters

- [[concepts/architectures/linear-layer|Linear layers]] are affine maps.
- [[concepts/architectures/attention|Attention]] uses matrix products between queries and keys.
- [[concepts/llm/embedding-retrieval|Embedding retrieval]] uses vector similarity.
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]] often finds lower-dimensional linear subspaces.

## Checks

- What are the tensor shapes?
- Which dimensions are batch, sequence, feature, head, node, or coordinate axes?
- Is a transformation linear, affine, normalized, or nonlinear?
- Is similarity a dot product, cosine similarity, distance, or learned score?
- Are tensors written in batch-first, sequence-first, or channel-first order?
- Is the operation preserving rank, reducing dimension, or mixing axes?
- Are coordinates, features, heads, and tokens being multiplied along the intended axes?

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
