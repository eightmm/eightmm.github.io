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

## Representation Choices

| Representation | Model Sees | Useful For | Main Risk |
| --- | --- | --- | --- |
| raw coordinates | atom/residue/point coordinates | coordinate prediction, refinement, equivariant models | arbitrary frame and missing elements |
| pairwise distances | $d_{ij}=\|x_i-x_j\|_2$ | invariant scalar models, contact maps | loses chirality and orientation |
| local frame features | coordinates relative to anchors | residue or pocket local geometry | frame construction may leak unavailable context |
| graph from coordinates | nodes plus distance/contact edges | molecular and protein graph models | cutoff or neighbor rule changes input |
| grid or voxel | discretized density or occupancy | CNN-style spatial models | resolution and alignment artifacts |
| surface or mesh | geometric boundary | pockets, shape complementarity | surface generation parameters affect claims |

## Coordinate Contract

For reusable notes, state:

$$
\mathcal{X}
=
(X, A, M, c)
$$

where $X\in\mathbb{R}^{N\times 3}$ stores coordinates, $A$ maps rows to atoms/residues/points, $M$ is a validity mask, and $c$ stores source context such as experimental, predicted, docked, generated, or simulated.

The transform rule should be explicit:

$$
X' = XR^\top + \mathbf{1}t^\top
$$

Scalar outputs such as affinity, energy, class probability, or rank should be invariant:

$$
f(X') = f(X)
$$

Coordinate, force, direction, or vector outputs should be equivariant:

$$
g(X') = g(X)R^\top
$$

## Structure-Based Risks

- A pocket cropped around a known ligand pose may leak deployment-unavailable information.
- Aligning structures to a template can hide whether the model is truly invariant.
- Missing atoms, unresolved loops, alternate locations, protonation states, and chain selection change the input object.
- Symmetric atoms or repeated residues require explicit mapping before coordinate losses or RMSD.
- Reflection invariance can be wrong for chiral molecules and protein structures.

## Practical Checks

- Are coordinates experimental, predicted, simulated, docked, or generated?
- Is the task invariant to rotation, translation, and reflection?
- Are missing atoms, alternate conformations, or low-confidence regions handled?
- Does the model use coordinates, distances, graphs, surfaces, grids, or mixed representations?
- Does evaluation check geometric validity, not only scalar metrics?
- Is the atom/residue/point mapping fixed before computing coordinate losses?
- Is pocket or crop definition available at deployment?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[papers/architectures/pointnet|PointNet]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
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
