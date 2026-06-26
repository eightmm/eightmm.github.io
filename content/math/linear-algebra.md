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
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/embedding|Embedding]]

## AI Connections

- Embedding similarity uses dot products, cosine similarity, and vector norms.
- Attention scores use matrix products before softmax.
- PCA, low-rank structure, and representation diagnostics use eigenvectors and SVD.
- Linear probes test whether a representation already contains task-relevant information.

## Checks

- What are the shapes of vectors and matrices?
- Is a transformation mixing features, tokens, nodes, or channels?
- Is similarity normalized or scale-sensitive?
- Is a low-rank approximation a model assumption or only an analysis tool?

## Related

- [[math/index|Math]]
- [[ai/architectures|Architectures]]
- [[concepts/learning/linear-probing|Linear probing]]
