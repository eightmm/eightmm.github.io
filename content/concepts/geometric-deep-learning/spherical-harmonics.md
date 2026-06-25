---
title: Spherical Harmonics
tags:
  - geometric-deep-learning
  - geometry
  - spherical-harmonics
---

# Spherical Harmonics

Spherical harmonics are basis functions on the sphere used to represent angular structure.

They are commonly written as:

$$
Y_\ell^m(\theta,\phi)
$$

where $\ell$ is the degree and $m$ is the order. Higher $\ell$ captures higher-frequency angular structure.

## Why It Matters

- They encode directional information in rotation-aware neural networks.
- They provide a practical bridge between 3D geometry and [[concepts/geometric-deep-learning/irreducible-representation|irreducible representations]].
- In molecular models, they can describe relative directions between atoms or residues.

## Checks

- Are angular features actually needed, or are distances sufficient?
- What maximum degree is used, and what compute cost does it add?
- Are directions normalized and numerically stable?
- Does the model combine radial and angular information cleanly?

## Related

- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor field network]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[entities/structure|Structure]]
