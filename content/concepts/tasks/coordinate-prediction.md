---
title: Coordinate Prediction
tags:
  - tasks
  - coordinates
  - geometric-deep-learning
---

# Coordinate Prediction

Coordinate prediction outputs points, structures, poses, trajectories, or coordinate updates. It is common in protein structure prediction, pose generation, molecular conformer generation, object tracking, robotics, and physical simulation.

For $n$ objects in $d$ dimensions:

$$
\hat{X}
=
f_\theta(x),
\qquad
\hat{X}\in\mathbb{R}^{n\times d}
$$

where each row $\hat{x}_i$ is a predicted coordinate.

## Symmetry

Many coordinate tasks should be equivariant under rotations and translations:

$$
f_\theta(g\cdot x)
=
g\cdot f_\theta(x)
$$

For a rigid transform $g=(R,t)$:

$$
x_i'
=
R x_i + t
$$

If the target is a scalar such as affinity or energy, it is usually invariant. If the target is coordinates, directions, velocities, forces, or coordinate updates, it is usually equivariant.

## Losses

Raw coordinate loss is:

$$
\mathcal{L}_{\mathrm{coord}}
=
\frac{1}{n}\sum_{i=1}^{n}
\lVert \hat{x}_i - x_i \rVert_2^2
$$

For structures where absolute frame is arbitrary, compare after alignment:

$$
\operatorname{RMSD}
=
\sqrt{
\frac{1}{n}
\min_{R,t}
\sum_{i=1}^{n}
\lVert R\hat{x}_i+t-x_i\rVert_2^2
}
$$

Distance-based objectives compare pairwise geometry:

$$
d_{ij}
=
\lVert x_i - x_j\rVert_2
$$

This can be invariant to global rotations and translations, but may lose chirality or orientation information.

## Checks

- What coordinate frame is used?
- Is the task invariant or equivariant under translation, rotation, reflection, or permutation?
- Are units fixed, e.g. pixels, meters, angstroms, or nanometers?
- Are missing atoms, residues, landmarks, or frames handled explicitly?
- Is alignment part of training, evaluation, both, or neither?
- Does preprocessing use information that would be unavailable at inference time?

## Related

- [[concepts/tasks/localization|Localization]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
