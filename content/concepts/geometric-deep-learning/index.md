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

## Decision Pattern

For a geometric model, decide the contract before choosing the architecture:

$$
(\text{object}, \text{target}, \text{group}, \text{split})
\rightarrow
\text{feature and readout design}
$$

- Object: molecule, protein, protein-ligand complex, point cloud, graph, or structure.
- Target: scalar property, ranking, distance, vector field, coordinate update, pose, or generated structure.
- Group: permutation, translation, rotation, reflection, SO(3), SE(3), or E(3).
- Split: ligand scaffold, protein family, complex pair, assay/source, or time when relevant.

The group choice should match both the data and the deployment setting. A symmetry enforced by preprocessing is only valid if the same information is available at inference time.

## Math Background

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/linear-algebra|Linear algebra]]

## Symmetry

- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/e3|E(3)]]

## Coordinates and Distances

- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]

## Representations

- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor Field Network]]

## Models and Operations

- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]

## Public-Safe Checks

- State the coordinate source: experimental, predicted, docked, generated, simulated, or conformer-generated.
- State whether chirality and stereochemistry are retained.
- State whether graph construction uses only deployment-available inputs.
- State whether scalar outputs are invariant and coordinate/vector outputs are equivariant.
- Avoid private structures, unpublished results, host paths, and internal benchmark names.

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[research/structure-based-ai/index|Structure-based modeling]]
