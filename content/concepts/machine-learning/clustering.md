---
title: Clustering
tags:
  - machine-learning
---

# Clustering

Clustering groups examples without using explicit labels. It is useful for exploration, data cleaning, representation analysis, and candidate grouping.

The core assumption is that the representation space contains meaningful neighborhood structure. If $z_i=\phi(x_i)$ is a learned embedding, clustering is really a statement about distances in $z$-space, not about the raw object $x_i$ itself.

k-means is the canonical example:

$$
\min_{\{\mu_k\}_{k=1}^{K}}
\sum_{i=1}^{n}
\min_{k}
\lVert x_i - \mu_k \rVert_2^2
$$

Here $\mu_k$ is a cluster centroid and each point is assigned to the nearest centroid.

Equivalently, with assignment variables $c_i\in\{1,\dots,K\}$:

$$
c_i
= \arg\min_k \lVert x_i-\mu_k\rVert_2^2
$$

$$
\mu_k
=
\frac{1}{|\{i:c_i=k\}|}
\sum_{i:c_i=k} x_i
$$

This alternating assignment-update view is useful because it shows why k-means prefers roughly spherical, similar-scale clusters.

## Common Methods

- k-means groups points around centroids.
- Hierarchical clustering builds nested groups.
- Gaussian mixture models cluster by a probabilistic mixture:

$$
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x;\mu_k,\Sigma_k)
$$

- Density-based methods group points in dense regions and can mark low-density points as noise.
- Spectral clustering builds a graph, then clusters using eigenvectors of a graph matrix.

## Distance Choice

The distance function is part of the model. Euclidean distance emphasizes absolute vector geometry, cosine distance emphasizes angle, and graph distance emphasizes connectivity.

For normalized embeddings, cosine similarity is:

$$
\operatorname{sim}(z_i,z_j)
=
\frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2}
$$

Changing $\phi(x)$ or the metric can change the cluster structure more than changing the clustering algorithm.

## Evaluation

Without labels, clustering quality is indirect. Common checks include cluster stability under resampling, nearest-neighbor inspection, and whether clusters explain downstream errors.

When labels are available for analysis, do not treat them as clustering targets unless the task is actually supervised. A cluster can be useful even if it does not align with a known label, and a label-aligned cluster can still be an artifact of leakage.

## Watch For

- Clusters are not ground truth labels.
- Results depend heavily on representation and distance metric.
- Visual clusters can be misleading after dimensionality reduction.
- Choosing $K$ from test-set behavior turns exploratory clustering into hidden model selection.
- High-dimensional distances can concentrate, making nearest-neighbor structure unstable.

## Related

- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/index|Evaluation]]
