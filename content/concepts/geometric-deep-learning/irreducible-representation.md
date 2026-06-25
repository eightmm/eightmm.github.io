---
title: Irreducible Representation
tags:
  - geometric-deep-learning
  - representation-theory
  - equivariance
---

# Irreducible Representation

An irreducible representation is a basic transformation block that cannot be decomposed into smaller independent representations.

For a group action represented by $D(g)$, a feature transforms as:

$$
h' = D(g)h
$$

Irreducible representations organize channels by transformation behavior, such as scalar or vector channels.

## Why It Matters

- Equivariant networks use irreducible representations to organize feature channels by transformation behavior.
- Scalars, vectors, and higher-order features can be tracked explicitly instead of mixed arbitrarily.
- This vocabulary clarifies how [[concepts/geometric-deep-learning/so3|SO(3)]] and [[concepts/geometric-deep-learning/e3|E(3)]] models handle symmetry.

## Checks

- Which feature types are present: scalars, vectors, or higher-order channels?
- Are operations allowed to mix representations only in symmetry-respecting ways?
- Does the architecture need high-order features, or are scalar/vector features enough?
- Are tensor products and nonlinearities the computational bottleneck?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
