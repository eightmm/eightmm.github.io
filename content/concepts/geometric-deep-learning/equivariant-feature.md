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

## Feature Map

| Feature | Transform Rule | Use |
| --- | --- | --- |
| coordinate $x_i$ | $x_i' = Rx_i+t$ | point position, pose, conformer, structure |
| relative vector $r_{ij}=x_j-x_i$ | $r_{ij}'=Rr_{ij}$ | direction-aware message passing |
| unit direction $u_{ij}$ | $u_{ij}'=Ru_{ij}$ | normalized geometric direction |
| force $F_i$ | $F_i'=RF_i$ | dynamics, energy gradients |
| velocity $v_i$ | $v_i'=Rv_i$ | flow matching, coordinate ODE |
| tensor feature | $T'=\rho(R)T$ | higher-order angular information |

## Scalar-Vector Separation

A common pattern is to compute invariant scalar weights and apply them to equivariant directions:

$$
\Delta x_i
=
\sum_{j\in\mathcal{N}(i)}
a_{ij}\,(x_j-x_i),
\qquad
a_{ij}=\phi(h_i,h_j,\|x_i-x_j\|_2,e_{ij})
$$

If $a_{ij}$ is invariant and the direction rotates, then $\Delta x_i$ rotates. This gives a simple route to equivariant coordinate updates without putting absolute coordinates into scalar features.

## Readout Boundary

| Target | Correct Output Type |
| --- | --- |
| affinity, energy, probability, class, rank | invariant scalar |
| force, velocity, score field, displacement | equivariant vector |
| coordinates, pose, conformer, backbone | equivariant coordinates plus translation behavior |
| distance, contact, RMSD-like diagnostic | invariant scalar derived from coordinates |
| orientation-sensitive chirality | preserve orientation information; reflection behavior must be explicit |

## Checks

- What representation does each feature live in: scalar, vector, tensor, or higher-order irrep?
- Does the output need translation equivariance, rotation equivariance, or both?
- Are reflections valid for this domain, or does chirality require orientation-preserving transforms?
- Are equivariant features converted to invariant readouts before scalar prediction?
- Is every preprocessing step equivariant, or only the neural layer?
- Are coordinate augmentations applied consistently to input, target, vector labels, and masks?
- Are units and coordinate frames fixed before comparing models?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
