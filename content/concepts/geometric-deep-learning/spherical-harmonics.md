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

For a unit direction $\hat{r}\in S^2$, a spherical harmonic maps direction to an angular basis value:

$$
Y_\ell^m: S^2 \rightarrow \mathbb{C},
\qquad
-\ell \le m \le \ell
$$

The degree $\ell$ has $2\ell+1$ components. In equivariant neural networks, these components transform together under rotation:

$$
Y_\ell(R\hat{r})
=
D^\ell(R)Y_\ell(\hat{r})
$$

where $R\in SO(3)$ and $D^\ell(R)$ is the Wigner-D matrix for degree $\ell$.

## Radial and Angular Split

For two points $x_i,x_j\in\mathbb{R}^3$, define:

$$
r_{ij}=x_j-x_i,\qquad
d_{ij}=\|r_{ij}\|_2,\qquad
\hat{r}_{ij}=\frac{r_{ij}}{\|r_{ij}\|_2+\epsilon}
$$

Many 3D models separate distance and direction:

$$
\phi(r_{ij})
=
R_n(d_{ij})Y_\ell^m(\hat{r}_{ij})
$$

where $R_n$ is a radial basis and $Y_\ell^m$ is an angular basis. Distances are rotation-invariant; directions carry rotation-equivariant information.

## Why It Matters

- They encode directional information in rotation-aware neural networks.
- They provide a practical bridge between 3D geometry and [[concepts/geometric-deep-learning/irreducible-representation|irreducible representations]].
- In molecular models, they can describe relative directions between atoms or residues.
- They help decide whether a model needs only scalar distance features or higher-order angular features.

## Degree Choice

| Maximum Degree | Captures | Cost/Risk |
| --- | --- | --- |
| $\ell_{\max}=0$ | scalar radial information only | cannot represent direction-dependent interactions |
| $\ell_{\max}=1$ | vector-like directional structure | moderate cost, often enough for vector fields |
| $\ell_{\max}\ge 2$ | higher-order angular patterns | higher tensor-product and memory cost |

For docking, conformers, and protein-ligand interactions, angular features can matter when orientation, directionality, or local geometry is part of the claim. If the task only predicts an invariant scalar from a fixed graph, distances and scalar features may be sufficient.

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
