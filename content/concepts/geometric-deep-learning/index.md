---
title: Geometric Deep Learning
tags:
  - geometric-deep-learning
  - geometry
---

# Geometric Deep Learning

Geometric deep learning studies models that respect structure such as graphs, coordinates, symmetries, and manifolds.

The core question is how a model behaves under a transformation group $G$:

$$
f(g\cdot x) = \rho(g) f(x),
\qquad
g\in G
$$

Here $\rho(g)$ describes how the output representation should transform.

## Symmetry

- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/e3|E(3)]]

## Representations

- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor Field Network]]

## Models and Operations

- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[research/structure-based-ai/index|Structure-based AI]]
