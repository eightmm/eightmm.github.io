---
title: Dimensionality Reduction
tags:
  - machine-learning
---

# Dimensionality Reduction

Dimensionality reduction maps high-dimensional data into a smaller representation while trying to preserve useful structure.

For PCA, the projection matrix $W$ is chosen to preserve maximum variance:

$$
\max_{W^\top W = I}
\operatorname{Tr}(W^\top X^\top X W)
$$

Here $X$ is the centered data matrix and $W$ contains the projection directions.

## Uses

- Visualization.
- Noise reduction.
- Feature compression.
- Representation analysis.

## Common Methods

- PCA preserves directions of high variance.
- SVD gives a practical decomposition for low-rank structure.
- t-SNE and UMAP are often used for visualization.
- Autoencoders learn nonlinear compressed representations.

## Watch For

- A 2D visualization is not proof of separability.
- Neighborhood structure can be distorted.
- Fitting reduction before splitting data can leak information.

## Related

- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/machine-learning/clustering|Clustering]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
