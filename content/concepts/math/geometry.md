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

A collection of $n$ points is often written as a coordinate matrix:

$$
X
=
\begin{bmatrix}
x_1^\top \\
\vdots \\
x_n^\top
\end{bmatrix}
\in \mathbb{R}^{n\times d}
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

## Coordinate Frame

A coordinate frame is the convention used to assign numbers to geometry. The same object can have different coordinates under a different frame.

| Quantity | Frame behavior |
| --- | --- |
| pairwise distance $\|x_i-x_j\|_2$ | invariant to rotation and translation |
| angle between vectors | invariant to rotation |
| coordinate $x_i$ | changes with frame |
| displacement vector $x_i-x_j$ | rotates with frame |
| scalar field on points | often invariant or equivariant depending on target |

This distinction is central for structure models: coordinates are not arbitrary noise, but neither are they absolute labels unless the frame has physical meaning.

## Distance Matrix

For point sets, pairwise distances form a matrix:

$$
D_{ij}
=
\lVert x_i-x_j\rVert_2
$$

The distance matrix is invariant to rigid transformations:

$$
D(RX+\mathbf{1}t^\top)=D(X)
$$

It loses orientation and chirality information, but it is often useful for contact maps, molecular geometry, and structure comparison.

## Angles and Inner Products

Angles can be computed from normalized vectors:

$$
\cos\theta
=
\frac{u^\top v}{\|u\|_2\|v\|_2}
$$

For three points $a,b,c$, the angle at $b$ uses:

$$
u=a-b,\quad v=c-b
$$

This is why geometric features often use distances, bond angles, torsion angles, local frames, or relative coordinates rather than absolute coordinates.

## Key Ideas

- Points, vectors, distances, angles, frames, and transformations are the core objects.
- Some targets are geometric scalars, such as distance, energy, score, or affinity.
- Some targets are geometric objects, such as coordinates, directions, fields, or frames.
- If a task is coordinate-dependent, ask which transformations should preserve the answer.
- Geometry is the foundation; [[concepts/geometric-deep-learning/index|geometric deep learning]] is the neural modeling layer built on top.
- Distance-only representations are invariant but may discard orientation-sensitive information.
- Frame-dependent outputs must specify how they transform.

## Practical Checks

- What is the geometric object: point cloud, graph with coordinates, surface, grid, curve, or manifold?
- Which quantities are invariant to coordinate transforms?
- Which quantities should rotate, translate, or reflect with the input?
- Is the coordinate frame arbitrary, experimentally meaningful, or introduced by preprocessing?
- Does the representation use absolute coordinates, relative coordinates, distances, or local frames?
- Is chirality or orientation important for the task?

## Related

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[entities/structure|Structure]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
