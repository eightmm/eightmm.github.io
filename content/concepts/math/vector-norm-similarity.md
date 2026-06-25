---
title: Vector Norm and Similarity
tags:
  - math
  - linear-algebra
  - representation-learning
---

# Vector Norm and Similarity

Vector norms and similarity functions define how embeddings, features, gradients, and coordinates are compared. They appear in [[concepts/architectures/attention|Attention]], retrieval, contrastive learning, clustering, docking scores, and evaluation.

For vectors $x,y\in\mathbb{R}^d$, the dot product is:

$$
x^\top y
=
\sum_{i=1}^{d} x_i y_i
$$

The Euclidean norm is:

$$
\lVert x\rVert_2
=
\sqrt{x^\top x}
=
\sqrt{\sum_{i=1}^{d}x_i^2}
$$

The distance induced by this norm is:

$$
d_2(x,y)
=
\lVert x-y\rVert_2
$$

Cosine similarity normalizes away vector magnitude:

$$
\operatorname{cos}(x,y)
=
\frac{x^\top y}
{\lVert x\rVert_2\lVert y\rVert_2}
$$

## Common Choices

- Dot product: magnitude and direction both affect similarity.
- Cosine similarity: direction matters more than magnitude.
- Euclidean distance: absolute position in feature space matters.
- Squared distance: convenient for gradients and least-squares objectives.
- Learned score: a model replaces a fixed geometry with task-specific scoring.

## In Attention

Scaled dot-product attention uses dot products between queries and keys:

$$
s_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}
$$

The $\sqrt{d_k}$ factor keeps logits from growing too large as the key dimension increases.

## In Representation Learning

If embeddings are normalized:

$$
\tilde{x}
=
\frac{x}{\lVert x\rVert_2}
$$

then dot product and cosine similarity coincide:

$$
\tilde{x}^\top \tilde{y}
=
\operatorname{cos}(x,y)
$$

This matters for retrieval, clustering, contrastive losses, and nearest-neighbor evaluation.

## Checks

- Are embeddings normalized before similarity is computed?
- Is the score a dot product, cosine similarity, distance, or learned function?
- Does vector magnitude carry meaningful confidence, frequency, or scale information?
- Is the metric aligned with the training loss and evaluation metric?
- Are features on comparable scales before distance-based methods are used?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/machine-learning/clustering|Clustering]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/evaluation/metric|Metric]]
