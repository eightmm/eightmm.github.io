---
title: SE(3)
tags:
  - geometric-deep-learning
  - symmetry
  - se3
---

# SE(3)

SE(3) is the group of 3D rotations and translations that preserve distances and orientation.

An SE(3) transform is:

$$
x' = Rx + t,
\qquad
R\in SO(3)
$$

Unlike E(3), it excludes reflections.

Composition is:

$$
(R_2,t_2)\circ(R_1,t_1)
=
(R_2R_1,\ R_2t_1+t_2)
$$

and the inverse is:

$$
(R,t)^{-1}
=
(R^\top,\ -R^\top t)
$$

Pairwise relative vectors transform as:

$$
(x_j'-x_i') = R(x_j-x_i)
$$

so their norms are invariant and their directions are equivariant.

## Why It Matters

- Many molecular structures can be moved or rotated in space without changing their physical identity.
- SE(3)-equivariant models are useful when predictions include vectors or coordinates.
- The distinction between SE(3), [[concepts/geometric-deep-learning/e3|E(3)]], and [[concepts/geometric-deep-learning/so3|SO(3)]] affects what symmetries are enforced.

## Output Contracts

- Scalar: $f(RX+\mathbf{1}t^\top)=f(X)$.
- Vector field: $V(RX+\mathbf{1}t^\top)=V(X)R^\top$ under row-vector convention.
- Coordinates: $F(RX+\mathbf{1}t^\top)=F(X)R^\top+\mathbf{1}t^\top$.

State the convention so equations and code agree.

## Failure Modes

- Centering removes translation but hides whether deployment uses the same centering rule.
- Reflection-sensitive chemistry is modeled with E(3) when SE(3) is intended.
- Vector labels such as forces are not transformed together with augmented coordinates.
- A bound ligand pose defines the coordinate frame for a task where the ligand pose is unknown at inference time.

## Checks

- Are translations and rotations both irrelevant to the input coordinate frame?
- Should reflections be excluded because orientation or chirality matters?
- Are outputs invariant scalars or SE(3)-equivariant vectors/coordinates?
- Does centering or alignment accidentally remove part of the modeling problem?
- Are coordinate units and row/column vector conventions explicit?
- Does preprocessing use only deployment-available information?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[entities/structure|Structure]]
