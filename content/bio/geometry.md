---
title: Geometry
aliases:
  - bio-ai/geometry
tags:
  - bio
  - geometry
---


# Geometry

Geometry connects Bio to graph, coordinate, symmetry, and equivariant modeling. Molecules and protein complexes are not only strings or graphs; many tasks depend on distances, angles, frames, and valid coordinate transformations.

$$
F(RX + t) = R F(X) + t
$$

This is the basic shape of equivariance for coordinate-valued outputs.

## Math and Geometry

- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/modalities/graph|Graph]]

## Geometric Deep Learning

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor field network]]

## Structure Tasks

- [[concepts/tasks/coordinate-prediction|Coordinate prediction]]
- [[concepts/tasks/graph-prediction|Graph prediction]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]

## Checks

- Which outputs should be invariant and which should be equivariant?
- Are coordinates centered, aligned, or frame-dependent?
- Are edges constructed only from inputs available at inference time?
- Are chirality, stereochemistry, units, and atom/residue indexing preserved?

## Related

- [[bio/structure-based-ai|Structure-based AI]]
- [[ai/architectures|Architectures]]
- [[concepts/architectures/gnn|Graph neural networks]]
