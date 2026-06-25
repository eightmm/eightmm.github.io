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

Matrix multiplication:

$$
C_{ij} = \sum_k A_{ik}B_{kj}
$$

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

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
