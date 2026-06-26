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

For invariant scalar outputs such as affinity or class probability, the usual requirement is:

$$
f(RX+t) = f(X)
$$

where $X$ is a coordinate set, $R$ is a rotation, and $t$ is a translation.

## Math and Geometry

- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/modalities/graph|Graph]]

## Geometric Deep Learning

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor field network]]

## Target Type Map

| Target | Symmetry Requirement | Examples |
| --- | --- | --- |
| Scalar | invariant | energy, affinity, class probability, ranking score |
| Vector / direction | equivariant | force, displacement, velocity, coordinate update |
| Coordinate set | equivariant up to permutation and rigid motion | ligand pose, atom coordinates, residue coordinates |
| Graph relation | often permutation equivariant/invariant | contact map, interaction edge, bond graph |

## Coordinate Features

| Feature | Formula | Use |
| --- | --- | --- |
| Distance | $d_{ij}=\lVert x_i-x_j\rVert_2$ | invariant edge feature |
| Direction | $u_{ij}=(x_j-x_i)/(d_{ij}+\epsilon)$ | equivariant message direction |
| Centered coordinate | $\tilde{x}_i=x_i-\frac{1}{N}\sum_j x_j$ | translation handling |
| Pairwise radial basis | $\psi(d_{ij})$ | smooth distance embedding |

## Structure Tasks

- [[concepts/tasks/coordinate-prediction|Coordinate prediction]]
- [[concepts/tasks/graph-prediction|Graph prediction]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]

## Checks

- Which outputs should be invariant and which should be equivariant?
- Are coordinates centered, aligned, or frame-dependent?
- Does the coordinate modeling contract match the claimed output and metric?
- Are edges constructed only from inputs available at inference time?
- Are chirality, stereochemistry, units, and atom/residue indexing preserved?

## Related

- [[bio/structure-based-ai|Structure-based modeling]]
- [[ai/architectures|Architectures]]
- [[concepts/architectures/gnn|Graph neural networks]]
