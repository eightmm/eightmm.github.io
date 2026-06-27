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

## Transformation Rules

| Object | Rule | Interpretation |
| --- | --- | --- |
| Translation | $x_i' = x_i + t$ | origin should not matter for many structure tasks |
| Rotation | $x_i' = R x_i$ | orientation should not change scalar labels |
| Rigid motion | $x_i' = R x_i + t$ | coordinates move together as one body |
| Permutation | $X' = P X$ | node/residue/atom order should not change the object |
| Invariant output | $f(g\cdot x)=f(x)$ | scalar target stays the same |
| Equivariant output | $F(g\cdot x)=\rho(g)F(x)$ | output transforms predictably with input |

For distances:

$$
d_{ij}
=
\lVert x_i-x_j\rVert_2
=
\lVert (Rx_i+t)-(Rx_j+t)\rVert_2
$$

so pairwise distances are invariant to rigid motions.

## Invariance vs Equivariance

Use invariance when the output should not change under a transformation:

$$
f(g\cdot x)=f(x)
$$

Use equivariance when the output should transform in a known way:

$$
F(g\cdot x)=\rho(g)F(x)
$$

where $\rho(g)$ is the representation of the transformation on the output space. For scalar labels, $\rho(g)$ is often the identity. For coordinate or vector outputs, $\rho(g)$ can be a rotation matrix or a rigid-motion action.

## Coordinate Sets

For a coordinate matrix:

$$
X =
\begin{bmatrix}
x_1^\top \\
\cdots \\
x_N^\top
\end{bmatrix}
\in \mathbb{R}^{N\times 3}
$$

a rigid motion acts as:

$$
X' = X R^\top + \mathbf{1}t^\top
$$

Permutation of atom or residue order acts as:

$$
X' = P X
$$

A coordinate model that predicts updated coordinates should usually satisfy:

$$
F(PXR^\top+\mathbf{1}t^\top)
=
P F(X) R^\top + \mathbf{1}t^\top
$$

This equation is a compact way to check whether a structure model respects both indexing and geometry.

## Groups and Coordinates

- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]

## Target Map

| Target | Expected Symmetry | Examples |
| --- | --- | --- |
| Class / affinity / energy | invariant | property prediction, binding score |
| Coordinate update | equivariant | pose generation, structure refinement |
| Force / vector field | equivariant | molecular dynamics, flow matching in coordinates |
| Graph relation | permutation equivariant or invariant | contact map, interaction graph |

## AI Connections

- CNNs encode translation-related inductive bias.
- GNNs handle permutation-sensitive relational data through message passing.
- Equivariant networks are important for molecules, proteins, and coordinate prediction.
- Docking and pose generation need geometry validity, not only scalar scores.

## Model Family Map

| Model Family | Built-In Symmetry | Typical Use |
| --- | --- | --- |
| CNN | translation equivariance on grids | images, local spatial patterns |
| GNN | permutation equivariance over nodes | molecules, residues, relational data |
| Graph Transformer | permutation-aware attention with graph bias | long-range graph interactions |
| E(3)/SE(3)-equivariant model | rigid-motion equivariance | coordinate prediction, forces, poses |
| Set model | permutation invariance/equivariance | unordered objects, pooled representations |

The model family should match the symmetry of the object and target, not only the dataset format.

## Checks

- Which transformations preserve the label?
- Which outputs should rotate or translate with the input?
- Is the coordinate frame arbitrary or physically meaningful?
- Is the model using absolute coordinates where relative geometry would be safer?
- Are atom, residue, or node permutations handled explicitly?
- Does the evaluation metric respect the same symmetry as the model target?

## Related

- [[math/index|Math]]
- [[molecular-modeling/geometry|Computational Biology geometry]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
