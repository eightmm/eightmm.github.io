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

## Group Action

A group action must be compatible with the group operation:

$$
e\cdot x = x
$$

and:

$$
(g_1\circ g_2)\cdot x
=
g_1\cdot(g_2\cdot x)
$$

This is what lets a model reason about transformations consistently. If preprocessing breaks this relationship, the claimed symmetry is no longer exact.

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

## Representation on Outputs

The output may transform differently from the input.

| Task output | Desired behavior |
| --- | --- |
| class label | invariant under allowed transformations |
| scalar energy or score | invariant under rigid motion |
| coordinate prediction | equivariant under rotation/translation |
| vector field or force | rotates with the input frame |
| graph node labels | permutes with node ordering |

For example, if $x_i\in\mathbb{R}^3$ are coordinates and $R$ is a rotation, a coordinate-predicting model should satisfy:

$$
f(RX+t)=Rf(X)+t
$$

while an energy model should satisfy:

$$
E(RX+t)=E(X)
$$

## Exact, Approximate, Broken

Not every useful symmetry is exact.

| Symmetry status | Meaning | Example |
| --- | --- | --- |
| exact | task definition should respect it | rigid-motion invariance of molecular energy |
| approximate | data mostly respects it but exceptions exist | image translation with boundary effects |
| intentionally broken | external frame matters | gravity direction, lab frame, camera view |
| preprocessing-induced | pipeline chooses a canonical frame | aligned structures or cropped images |

The modeling choice should match the task, not only the data format.

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
- What is the group action on the input and representation on the output?
- Does augmentation encourage symmetry, or does architecture enforce it?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/architectures/gnn|Graph neural networks]]
