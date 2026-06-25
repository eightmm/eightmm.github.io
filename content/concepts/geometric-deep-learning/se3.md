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

## Why It Matters

- Many molecular structures can be moved or rotated in space without changing their physical identity.
- SE(3)-equivariant models are useful when predictions include vectors or coordinates.
- The distinction between SE(3), [[concepts/geometric-deep-learning/e3|E(3)]], and [[concepts/geometric-deep-learning/so3|SO(3)]] affects what symmetries are enforced.

## Checks

- Are translations and rotations both irrelevant to the input coordinate frame?
- Should reflections be excluded because orientation or chirality matters?
- Are outputs invariant scalars or SE(3)-equivariant vectors/coordinates?
- Does centering or alignment accidentally remove part of the modeling problem?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[entities/structure|Structure]]
