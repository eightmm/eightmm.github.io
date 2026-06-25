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

## Why It Matters

- Rotational symmetry is central to models over coordinates, directions, and local frames.
- Spherical features and tensor features are often organized by how they transform under SO(3).
- SO(3) is a building block for [[concepts/geometric-deep-learning/se3|SE(3)]] methods that also handle translation.

## Checks

- Is the task centered so rotations are the main transformation?
- Are local frames, directions, and angular features represented consistently?
- Does the model require spherical harmonics or a simpler vector representation?
- Are scalar outputs invariant after rotation-dependent features are processed?

## Related

- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor field network]]
