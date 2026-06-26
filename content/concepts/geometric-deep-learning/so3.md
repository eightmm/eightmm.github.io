---
title: SO(3)
tags:
  - geometric-deep-learning
  - symmetry
  - so3
---

# SO(3)

SO(3) is the group of 3D rotations around the origin.

The group is:

$$
SO(3)=\{R\in \mathbb{R}^{3\times 3}\mid R^\top R=I,\ \det(R)=1\}
$$

It contains rotations but not translations or reflections.

The group action on a point or vector is:

$$
x' = Rx
$$

Composition and inverse are:

$$
R_3 = R_2R_1,
\qquad
R^{-1}=R^\top
$$

Distances to the origin and angles are preserved:

$$
\lVert Rx\rVert_2 = \lVert x\rVert_2,
\qquad
(Ru)^\top(Rv)=u^\top v
$$

## Relation to Features

- Scalar features are invariant under SO(3).
- Vector features transform as $v'=Rv$.
- Higher-order tensor features transform under higher-dimensional representations.
- Spherical harmonics and irreducible representations organize angular information by rotation behavior.

## Why It Matters

- Rotational symmetry is central to models over coordinates, directions, and local frames.
- Spherical features and tensor features are often organized by how they transform under SO(3).
- SO(3) is a building block for [[concepts/geometric-deep-learning/se3|SE(3)]] methods that also handle translation.

## Failure Modes

- Using SO(3) alone when translation is also arbitrary.
- Treating reflections as rotations; reflections have determinant $-1$ and are not in SO(3).
- Aligning every sample to a canonical frame and accidentally injecting target-specific information.

## Checks

- Is the task centered so rotations are the main transformation?
- Are local frames, directions, and angular features represented consistently?
- Does the model require spherical harmonics or a simpler vector representation?
- Are scalar outputs invariant after rotation-dependent features are processed?
- Are reflections intentionally excluded?
- Is the origin meaningful, or should the model use SE(3) instead?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor field network]]
