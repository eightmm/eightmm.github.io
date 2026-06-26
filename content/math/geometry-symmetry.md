---
title: Geometry and Symmetry
tags:
  - math
  - geometry
  - symmetry
---

# Geometry and Symmetry

Geometry and symmetry explain what should stay the same, and what should transform predictably, when coordinates are moved, rotated, permuted, or re-indexed.

$$
f(g \cdot x) = g \cdot f(x)
$$

This is the abstract form of equivariance.

## Core Notes

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]

## Groups and Coordinates

- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]

## AI Connections

- CNNs encode translation-related inductive bias.
- GNNs handle permutation-sensitive relational data through message passing.
- Equivariant networks are important for molecules, proteins, and coordinate prediction.
- Docking and pose generation need geometry validity, not only scalar scores.

## Checks

- Which transformations preserve the label?
- Which outputs should rotate or translate with the input?
- Is the coordinate frame arbitrary or physically meaningful?
- Is the model using absolute coordinates where relative geometry would be safer?

## Related

- [[math/index|Math]]
- [[bio/geometry|Bio geometry]]
- [[concepts/architectures/gnn|Graph neural networks]]
