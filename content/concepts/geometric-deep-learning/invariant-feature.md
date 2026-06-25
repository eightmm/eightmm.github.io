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

## Checks

- Which group should the feature be invariant to: translation, rotation, reflection, permutation, or sequence shift?
- Is the target also invariant to that group?
- Does the feature discard information needed for the task, such as chirality or orientation?
- Is the invariant feature computed from public/generic inputs rather than deployment-unavailable context?

## Related

- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/modalities/3d-structure|3D structure]]
