---
title: Geometric Deep Learning
tags:
  - geometric-deep-learning
  - geometry
---

# Geometric Deep Learning

Geometric deep learning is the AI layer built on top of [[concepts/math/geometry|geometry]] and [[concepts/math/symmetry-group|symmetry groups]]. It studies models that respect structure such as graphs, coordinates, transformations, and manifolds.

This section should not be a pure math textbook. Use it as the bridge from math foundations to neural architectures for proteins, molecules, 3D structures, vision, and graph-structured data.

The core question is how a model behaves under a transformation group $G$:

$$
f(g\cdot x) = \rho(g) f(x),
\qquad
g\in G
$$

Here $\rho(g)$ describes how the output representation should transform.

## Math Background

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/linear-algebra|Linear algebra]]

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

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[research/structure-based-ai/index|Structure-based AI]]
