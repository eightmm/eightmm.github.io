---
title: Symmetry Group
tags:
  - math
  - symmetry
  - group-theory
---

# Symmetry Group

A symmetry group is a set of transformations that can be composed, inverted, and applied without changing the kind of object being studied. For machine learning, the important question is how data and predictions behave under these transformations.

A group $G$ has an operation $\circ$ such that:

$$
g_1,g_2\in G
\Rightarrow
g_1\circ g_2\in G
$$

It also has an identity element $e$ and inverse $g^{-1}$:

$$
e\circ g = g,
\qquad
g^{-1}\circ g = e
$$

A group action applies a group element to an object:

$$
g\cdot x
$$

Examples include rotating a point cloud, translating an image, permuting graph nodes, or reflecting a molecule.

## Invariance and Equivariance

An invariant function ignores the transformation:

$$
f(g\cdot x)=f(x)
$$

An equivariant function transforms its output predictably:

$$
f(g\cdot x)=\rho(g)f(x)
$$

where $\rho(g)$ is the representation of the transformation on the output space.

## Common Groups

- Permutation group: reorders set or graph elements.
- Translation group: shifts coordinates or grids.
- Rotation group: rotates vectors or structures.
- [[concepts/geometric-deep-learning/so3|SO(3)]]: 3D rotations.
- [[concepts/geometric-deep-learning/se3|SE(3)]]: 3D rotations and translations.
- [[concepts/geometric-deep-learning/e3|E(3)]]: 3D rotations, translations, and reflections.

## Practical Checks

- What transformation should not change the meaning of the input?
- Should the output be invariant, equivariant, or neither?
- Is the symmetry exact by construction, approximate in data, or broken by the task?
- Does preprocessing impose a coordinate frame that changes the symmetry assumption?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/architectures/gnn|Graph neural networks]]
