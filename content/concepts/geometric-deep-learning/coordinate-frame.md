---
title: Coordinate Frame
tags:
  - geometric-deep-learning
  - geometry
  - coordinates
---

# Coordinate Frame

A coordinate frame is the reference system used to describe positions, directions, and orientations. In 3D structure modeling, the same physical object can have many coordinate descriptions.

For points $x_i\in\mathbb{R}^3$, a rigid frame change is:

$$
x_i' = Rx_i + t
$$

where $R\in SO(3)$ is a rotation matrix and $t\in\mathbb{R}^3$ is a translation.

The object is unchanged, but its coordinates are different. A model should therefore specify whether its input features, hidden states, and outputs are frame-dependent or frame-independent.

## Common Frames

- Global frame: coordinates as stored in a structure file or simulation snapshot.
- Centered frame: coordinates shifted by a centroid, pocket center, or residue center.
- Local frame: coordinates expressed relative to an atom, residue, bond, or neighborhood.
- Learned frame: a model-predicted frame used for updates, alignment, or generation.

## Modeling Implications

Distances are invariant to rigid frame changes:

$$
\lVert x_i' - x_j' \rVert_2
=
\lVert R(x_i-x_j) \rVert_2
=
\lVert x_i-x_j \rVert_2
$$

Directions are equivariant:

$$
v_{ij}=x_j-x_i,
\qquad
v_{ij}'=Rv_{ij}
$$

If a model predicts coordinates, directions, forces, vector fields, or docking poses, the output should usually transform with the input frame. If it predicts class labels, energies, affinities, or rankings, the output should usually remain invariant.

## Pitfalls

- Do not let a preprocessing frame leak the answer. For example, a ligand-defined pocket frame can be unrealistic if the ligand pose is unavailable at deployment.
- Do not mix arbitrary global coordinates with task labels unless the task truly depends on that frame.
- Check whether reflection symmetry is allowed. Molecules can be sensitive to chirality, so $O(3)$ invariance may be too broad when $SE(3)$ equivariance is intended.
- Record whether coordinates are experimental, predicted, docked, generated, or simulated.

## Checks

- What frame are the input coordinates expressed in?
- Is the frame arbitrary, physically meaningful, or created by preprocessing?
- Which features are invariant scalars?
- Which features are equivariant vectors or tensors?
- Does train/test preprocessing use the same frame contract?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/modalities/3d-structure|3D structure]]
