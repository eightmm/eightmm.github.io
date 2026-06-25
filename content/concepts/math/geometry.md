---
title: Geometry
tags:
  - math
  - geometry
---

# Geometry

Geometry studies objects that have shape, distance, angle, position, and transformation structure. In this wiki, geometry is a math foundation used by geometric deep learning, molecular modeling, protein structures, robotics-style agents, and vision.

The basic Euclidean setting represents points as vectors:

$$
x \in \mathbb{R}^d
$$

The distance between two points is:

$$
d(x,y)=\lVert x-y\rVert_2
=
\sqrt{\sum_{i=1}^{d}(x_i-y_i)^2}
$$

An inner product defines angles:

$$
\langle x,y\rangle = x^\top y,
\qquad
\cos\theta =
\frac{x^\top y}{\lVert x\rVert_2\lVert y\rVert_2}
$$

## Transformations

Many geometry-aware models ask whether a quantity should change when the coordinate frame changes. A rigid transform in 3D is:

$$
x' = Rx + t
$$

where $R$ is a rotation matrix and $t$ is a translation vector.

Distances are preserved under rigid transforms:

$$
\lVert (Rx_i+t)-(Rx_j+t)\rVert_2
=
\lVert x_i-x_j\rVert_2
$$

This is the math reason why molecular identity should not depend on the arbitrary placement of a structure in a coordinate file.

## Key Ideas

- Points, vectors, distances, angles, frames, and transformations are the core objects.
- Some targets are geometric scalars, such as distance, energy, score, or affinity.
- Some targets are geometric objects, such as coordinates, directions, fields, or frames.
- If a task is coordinate-dependent, ask which transformations should preserve the answer.
- Geometry is the foundation; [[concepts/geometric-deep-learning/index|geometric deep learning]] is the neural modeling layer built on top.

## Practical Checks

- What is the geometric object: point cloud, graph with coordinates, surface, grid, curve, or manifold?
- Which quantities are invariant to coordinate transforms?
- Which quantities should rotate, translate, or reflect with the input?
- Is the coordinate frame arbitrary, experimentally meaningful, or introduced by preprocessing?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[entities/structure|Structure]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
