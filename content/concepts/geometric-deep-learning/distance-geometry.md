---
title: Distance Geometry
tags:
  - geometric-deep-learning
  - geometry
  - distance
---

# Distance Geometry

Distance geometry represents a structure through pairwise distances rather than absolute coordinates. It is useful when the coordinate frame is arbitrary but relative spatial relationships matter.

For coordinates $X=(x_1,\ldots,x_N)$, the distance matrix is:

$$
D_{ij} = \lVert x_i-x_j\rVert_2
$$

Under a rigid transform $x_i'=Rx_i+t$:

$$
D_{ij}' = \lVert x_i'-x_j'\rVert_2 = D_{ij}
$$

So $D$ is invariant to rotation and translation.

## Why It Matters

- Protein contact maps, residue distances, molecular conformers, and pocket-ligand contacts can be expressed through distances.
- Distance features are stable when absolute orientation is irrelevant.
- Distance matrices can support graph construction, edge features, and geometric constraints.

## Limits

Distance-only representations can lose information:

- Orientation can be missing.
- Chirality can be ambiguous if reflections are not constrained.
- Multiple coordinate sets can share similar distances under noise or missing entries.
- Long-range distance matrices scale as $O(N^2)$.

## From Distances to Coordinates

Classical multidimensional scaling starts from squared distances and builds a centered Gram matrix:

$$
B = -\frac{1}{2}J D^{(2)} J,
\qquad
J = I - \frac{1}{N}\mathbf{1}\mathbf{1}^\top
$$

If $B$ is positive semidefinite, coordinates can be recovered from its top eigenvectors:

$$
B = U\Lambda U^\top,
\qquad
X = U_k\Lambda_k^{1/2}
$$

This shows the link between distances, inner products, and coordinates.

## Practical Checks

- Are all distances observed, or only local/contact distances?
- Does the task need orientation, handedness, or local frames?
- Are distances used as labels, input features, graph edges, or constraints?
- Does the model output a valid geometry, not only accurate pairwise distances?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/modalities/3d-structure|3D structure]]
