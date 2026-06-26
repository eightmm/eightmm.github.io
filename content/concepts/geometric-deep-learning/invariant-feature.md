---
title: Invariant Feature
tags:
  - geometric-deep-learning
  - invariance
  - features
---

# Invariant Feature

An invariant feature does not change when the input is transformed by a symmetry group.

For a transformation $g\in G$ and feature map $\phi$:

$$
\phi(g\cdot x)=\phi(x)
$$

The feature can be used when the task should ignore that transformation.

## Examples

- Pairwise distance:

$$
d_{ij}=\lVert x_i-x_j\rVert_2
$$

- Angle between two vectors:

$$
\cos\theta =
\frac{u^\top v}{\lVert u\rVert_2\lVert v\rVert_2}
$$

- Norm of a vector:

$$
\lVert v\rVert_2
$$

- Graph-level readout after permutation-invariant pooling:

$$
h_G = \sum_{i=1}^{N} h_i
$$

## Uses

- Scalar property prediction.
- Binding affinity or energy prediction.
- Classification and ranking.
- Graph-level molecular readout.
- Distance/contact features for 3D structures.

## Feature Map

| Feature | Invariant To | Not Necessarily Preserved |
| --- | --- | --- |
| distance $\|x_i-x_j\|_2$ | translation and rotation | direction, chirality, absolute frame |
| angle $\angle ijk$ | translation and rotation | handedness unless signed orientation is included |
| dihedral magnitude | translation and rotation | sign can matter for stereochemistry |
| vector norm $\|v\|$ | rotation | vector direction |
| sorted neighbor distances | permutation within neighbors | identity of each neighbor |
| graph pooled state | node permutation | local attribution and rare-site signal |

## Invariance Claim Boundary

Invariant features are appropriate only when the target is also invariant:

$$
y(g\cdot x)=y(x)
$$

for the transformation group $G$. For scalar molecular properties this is usually true for rotations and translations. For coordinate updates, forces, velocities, and pose refinement, invariant features alone are insufficient unless they are combined with equivariant directions.

## Molecular Modeling Risks

| Risk | Example |
| --- | --- |
| chirality loss | using only pairwise distances can fail to distinguish mirror-image configurations |
| pose leakage | invariant distances computed from a known bound pose may be unavailable at inference |
| size leakage | sum features can encode molecule or protein size when size is a dataset artifact |
| context leakage | pocket features can be ligand-defined rather than deployment-defined |
| over-invariance | augmentation or feature design removes signal required by the label |

## Checks

- Which group should the feature be invariant to: translation, rotation, reflection, permutation, or sequence shift?
- Is the target also invariant to that group?
- Does the feature discard information needed for the task, such as chirality or orientation?
- Is the invariant feature computed from public/generic inputs rather than deployment-unavailable context?
- Is the invariant feature used as input, target, metric, or readout?
- Does the feature preserve atom/residue identity and mapping when required?

## Related

- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/modalities/3d-structure|3D structure]]
