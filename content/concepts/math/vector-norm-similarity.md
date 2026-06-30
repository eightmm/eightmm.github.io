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

More generally, the $p$-norm is:

$$
\lVert x\rVert_p
=
\left(\sum_{i=1}^{d}|x_i|^p\right)^{1/p}
$$

Common special cases are:

$$
\lVert x\rVert_1=\sum_i |x_i|,
\qquad
\lVert x\rVert_\infty=\max_i |x_i|
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

## Choice Table

| Function | Formula | Use when | Watch out |
| --- | --- | --- | --- |
| dot product | $x^\top y$ | magnitude carries signal | large-norm vectors dominate |
| cosine similarity | $\frac{x^\top y}{\|x\|_2\|y\|_2}$ | direction matters more than scale | loses confidence/frequency magnitude |
| Euclidean distance | $\|x-y\|_2$ | absolute coordinates are meaningful | sensitive to feature scaling |
| squared Euclidean | $\|x-y\|_2^2$ | smooth least-squares objective | grows fast for outliers |
| Manhattan distance | $\|x-y\|_1$ | sparse or coordinate-wise deviations | non-smooth at zero |
| learned score | $s_\theta(x,y)$ | task-specific similarity | harder to interpret and calibrate |

## Metric Conditions

A distance $d$ is a metric if it satisfies:

$$
d(x,y)\ge 0,\quad
d(x,y)=0\iff x=y
$$

$$
d(x,y)=d(y,x)
$$

$$
d(x,z)\le d(x,y)+d(y,z)
$$

Cosine similarity is a similarity score, not a distance metric by itself. Some systems convert it into a distance-like quantity, but metric assumptions should be checked before using nearest-neighbor indexes or evaluation metrics.

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

## Scaling and Normalization

Distance-based methods are sensitive to feature scale:

$$
d(Ax,Ay)
=
\lVert A(x-y)\rVert_2
$$

If $A$ rescales coordinates unevenly, the geometry changes. This is why feature normalization, whitening, learned projections, and embedding normalization can change retrieval or clustering behavior without changing the downstream model.

## Checks

- Are embeddings normalized before similarity is computed?
- Is the score a dot product, cosine similarity, distance, or learned function?
- Does vector magnitude carry meaningful confidence, frequency, or scale information?
- Is the metric aligned with the training loss and evaluation metric?
- Are features on comparable scales before distance-based methods are used?
- Does the nearest-neighbor backend assume metric properties?
- Is normalization removing useful magnitude information?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/machine-learning/clustering|Clustering]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/evaluation/metric|Metric]]
