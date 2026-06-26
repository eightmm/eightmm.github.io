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

The reduced representation is:

$$
Z = XW,
\qquad
Z\in\mathbb{R}^{n\times k},
\qquad
k \ll d
$$

where $n$ is the number of examples, $d$ is the original feature dimension, and $k$ is the reduced dimension.

## Uses

- Visualization.
- Noise reduction.
- Feature compression.
- Representation analysis.

## Common Methods

| Method | Preserves | Use With Care |
| --- | --- | --- |
| PCA | global linear variance directions | high variance may be nuisance variation |
| SVD | low-rank linear structure | scaling and centering change components |
| random projection | approximate distances under dimension reduction | interpretability is limited |
| t-SNE | local neighborhood visualization | global distances and cluster sizes are not reliable |
| UMAP | neighborhood graph visualization | hyperparameters strongly change the map |
| autoencoder | nonlinear reconstruction through a bottleneck | reconstruction quality is not task utility |

For SVD:

$$
X = U\Sigma V^\top,
\qquad
Z_k = XV_k
$$

where $V_k$ contains the top $k$ right singular vectors. PCA can be computed from this decomposition when $X$ is centered.

## Evaluation Uses

| Use | Correct Interpretation |
| --- | --- |
| embedding visualization | qualitative inspection only |
| feature compression | compare downstream metric at fixed train/validation/test split |
| denoising | verify that removed components are not task signal |
| representation analysis | inspect spectrum, rank, cluster stability, and nearest neighbors |
| preprocessing step | fit reducer on training data, then apply to validation/test |

## Watch For

- A 2D visualization is not proof of separability.
- Neighborhood structure can be distorted.
- Fitting reduction before splitting data can leak information.
- Choosing $k$ on the test set is model-selection leakage.
- Color-coded clusters may reflect source, batch, scaffold, or family rather than the intended label.
- UMAP/t-SNE plots should not be used as quantitative evidence without a task metric.

## Related

- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/clustering|Clustering]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
