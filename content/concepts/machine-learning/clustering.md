---
title: Clustering
tags:
  - machine-learning
---

# Clustering

Clustering groups examples without using explicit labels. It is useful for exploration, data cleaning, representation analysis, and candidate grouping.

k-means is the canonical example:

$$
\min_{\{\mu_k\}_{k=1}^{K}}
\sum_{i=1}^{n}
\min_{k}
\lVert x_i - \mu_k \rVert_2^2
$$

Here $\mu_k$ is a cluster centroid and each point is assigned to the nearest centroid.

## Common Methods

- k-means groups points around centroids.
- Hierarchical clustering builds nested groups.
- Density-based methods group points in dense regions.

## Watch For

- Clusters are not ground truth labels.
- Results depend heavily on representation and distance metric.
- Visual clusters can be misleading after dimensionality reduction.

## Related

- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/index|Evaluation]]
