---
title: Equivariant Feature
tags:
  - geometric-deep-learning
  - equivariance
  - features
---

# Equivariant Feature

An equivariant feature transforms predictably when the input is transformed.

For a transformation $g\in G$, input action $g\cdot x$, and output representation $\rho(g)$:

$$
\phi(g\cdot x)=\rho(g)\phi(x)
$$

For 3D rotations, a vector feature $v_i\in\mathbb{R}^3$ usually transforms as:

$$
v_i' = Rv_i
$$

where $R\in SO(3)$.

## Examples

- Coordinate:

$$
x_i' = Rx_i+t
$$

- Relative direction:

$$
u_{ij}=\frac{x_j-x_i}{\lVert x_j-x_i\rVert_2},
\qquad
u_{ij}'=Ru_{ij}
$$

- Coordinate update:

$$
x_i^{(t+1)} = x_i^{(t)} + \Delta x_i,
\qquad
\Delta x_i' = R\Delta x_i
$$

- Vector field, force, velocity, or score direction.

## Why It Matters

Equivariant features let a model predict geometry without learning arbitrary coordinate-frame rules from data. This is useful for structure prediction, docking pose generation, conformer generation, molecular dynamics surrogates, and 3D generative models.

## Checks

- What representation does each feature live in: scalar, vector, tensor, or higher-order irrep?
- Does the output need translation equivariance, rotation equivariance, or both?
- Are reflections valid for this domain, or does chirality require orientation-preserving transforms?
- Are equivariant features converted to invariant readouts before scalar prediction?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
