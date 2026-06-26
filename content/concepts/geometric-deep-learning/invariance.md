---
title: Invariance
tags:
  - geometric-deep-learning
  - symmetry
  - invariance
---

# Invariance

Invariance means a model output stays unchanged when an allowed transformation is applied to the input.

Formally:

$$
f(g\cdot x) = f(x)
$$

Invariant outputs are appropriate for scalar targets such as class labels, many energies, or global scores.

For 3D point coordinates:

$$
F(RX+\mathbf{1}t^\top) = F(X)
$$

when the target should not depend on global rotation or translation.

Distance is a basic invariant:

$$
d_{ij}
=
\lVert x_i-x_j\rVert_2,
\qquad
d_{ij}'
=
\lVert R(x_i-x_j)\rVert_2
=
d_{ij}
$$

Permutation-invariant graph or set pooling has the form:

$$
h_G = \operatorname{pool}\{h_i\}_{i=1}^{N},
\qquad
\operatorname{pool}(\pi h)=\operatorname{pool}(h)
$$

where $\pi$ reorders the elements.

## Why It Matters

- Molecular energies, class labels, and many scores should not change under global rotation or translation.
- Invariant features can summarize geometry without depending on an arbitrary coordinate frame.
- Choosing invariant outputs requires knowing the task target and its physical meaning.

## Target Examples

- Usually invariant: molecular property, binding-affinity scalar, class probability, ranking score, energy.
- Usually equivariant: coordinates, displacement, force, velocity, vector field, denoising direction.
- Context-dependent: pose quality, contact maps, distance matrices, and orientation-dependent interactions.

## Failure Modes

- Invariance removes chirality or orientation information that the task needs.
- Pooling mixes unrelated graphs or molecules because batch membership is ignored.
- A benchmark metric is invariant, but the deployed output is a coordinate or pose that needs equivariance.
- Invariance is created by alignment to unavailable information, causing coordinate-frame leakage.

## Checks

- Is the target truly invariant under rotation, translation, permutation, or reflection?
- Are invariant features discarding information needed for an equivariant output?
- Does pooling respect graph membership and atom or residue identity?
- Is invariance produced by architecture, augmentation, or preprocessing?
- Is reflection invariance valid for stereochemistry and chirality?
- Does the invariant readout match the actual decision metric?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[entities/structure|Structure]]
