---
title: 3D Structure
tags:
  - modalities
  - structure
  - geometry
---

# 3D Structure

3D structure is a modality where examples include coordinates, distances, orientations, surfaces, or spatial relationships. It is central to protein structures, ligand conformers, protein-ligand complexes, point clouds, and geometry-aware vision.

A coordinate representation is:

$$
X = (x_1,\ldots,x_N),
\qquad
x_i\in\mathbb{R}^3
$$

Many tasks should be invariant or equivariant to rigid transforms:

$$
x_i' = Rx_i+t
$$

where $R$ is a rotation matrix and $t$ is a translation.

Pairwise distances are often used as invariant features:

$$
d_{ij}=\lVert x_i-x_j\rVert_2
$$

## Key Ideas

- The absolute coordinate frame is often arbitrary; relative geometry usually carries the physical meaning.
- Coordinates can be represented directly, converted into distances, voxelized, or converted into graphs.
- Some outputs should be invariant scalars; others should be equivariant coordinates or vectors.
- Preprocessing choices such as centering, alignment, cropping, or pocket selection can change the task.
- Chirality and reflection symmetry need special care for molecular settings.

## Practical Checks

- Are coordinates experimental, predicted, simulated, docked, or generated?
- Is the task invariant to rotation, translation, and reflection?
- Are missing atoms, alternate conformations, or low-confidence regions handled?
- Does the model use coordinates, distances, graphs, surfaces, grids, or mixed representations?
- Does evaluation check geometric validity, not only scalar metrics?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[entities/structure|Structure]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/sbdd/pose-quality|Pose quality]]
