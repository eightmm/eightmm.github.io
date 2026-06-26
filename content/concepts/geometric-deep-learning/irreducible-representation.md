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

For 3D rotations, irreducible representations of $SO(3)$ are indexed by degree $\ell$:

$$
h_\ell' = D^\ell(R)h_\ell,
\qquad
R\in SO(3)
$$

The channel group $h_\ell$ has dimension $2\ell+1$. Common cases are:

| Degree | Feature Type | Transform Behavior |
| --- | --- | --- |
| $\ell=0$ | scalar | invariant under rotation |
| $\ell=1$ | vector-like feature | rotates like a 3D direction |
| $\ell\ge2$ | higher-order tensor feature | carries angular patterns beyond vectors |

An equivariant model keeps these feature types separate unless an operation is allowed by the group representation.

## Mixing Rule

If a layer combines two irreducible features, the output degrees are constrained by angular momentum coupling:

$$
\ell_{\mathrm{out}}
\in
\{|\ell_1-\ell_2|,\ldots,\ell_1+\ell_2\}
$$

This is why tensor products and Clebsch-Gordan coefficients appear in many $SO(3)$-equivariant architectures. The purpose is not mathematical decoration; it prevents arbitrary channel mixing that would break equivariance.

## Invariant Readout

If the final target is a scalar such as energy, affinity, class probability, or ranking score, the final readout usually needs $\ell=0$ output:

$$
\hat{y} = \rho(h_{\ell=0})
$$

If the target is a coordinate update, force, direction, or vector field, the output should transform equivariantly, often as $\ell=1$:

$$
\hat{v}' = R\hat{v}
$$

## Why It Matters

- Equivariant networks use irreducible representations to organize feature channels by transformation behavior.
- Scalars, vectors, and higher-order features can be tracked explicitly instead of mixed arbitrarily.
- This vocabulary clarifies how [[concepts/geometric-deep-learning/so3|SO(3)]] and [[concepts/geometric-deep-learning/e3|E(3)]] models handle symmetry.
- It makes the output contract explicit: scalar outputs should be invariant, while vector or coordinate outputs should transform predictably.

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
