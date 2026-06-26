---
title: Equivariance
tags:
  - geometric-deep-learning
  - symmetry
  - equivariance
---

# Equivariance

Equivariance means a model output transforms predictably when the input is transformed.

Formally, for a transformation $g$:

$$
f(g \cdot x) = g \cdot f(x)
$$

The output changes in the same structured way as the input.

More generally, inputs and outputs can transform under different representations:

$$
f(\rho_{\mathrm{in}}(g)x)
=
\rho_{\mathrm{out}}(g)f(x)
$$

Here $\rho_{\mathrm{in}}(g)$ describes how the input changes and $\rho_{\mathrm{out}}(g)$ describes how the output should change.

For 3D coordinates:

$$
x_i' = Rx_i + t,
\qquad
\hat{x}_i' = R\hat{x}_i + t
$$

where $R\in SO(3)$ and $t\in\mathbb{R}^3$. A coordinate predictor should satisfy:

$$
F(RX+\mathbf{1}t^\top)
=
RF(X)+\mathbf{1}t^\top
$$

with the same point ordering.

## Why It Matters

- Coordinate predictions should rotate or translate with the input structure.
- Equivariance reduces the burden of learning symmetries from data alone.
- Many geometric neural networks combine equivariant internal features with invariant task outputs.

## Common Groups

- Translation: shifting all coordinates by the same $t$.
- Rotation: applying $R\in SO(3)$.
- Rigid motion: applying $(R,t)\in SE(3)$.
- Rigid motion with reflection: applying $(R,t)$ with $R\in O(3)$, often called E(3).
- Permutation: reordering atoms, residues, nodes, or set elements.

## Failure Modes

- The model is equivariant to rotations but not translations because absolute coordinates enter hidden features.
- The architecture is equivariant but preprocessing uses a ligand-defined frame unavailable at deployment.
- Reflection equivariance is enforced even though chirality should be preserved.
- A scalar target is treated as equivariant, or a coordinate target is reduced to invariant features too early.

## Checks

- What transformation group is being considered?
- Which model outputs should transform, and which should remain invariant?
- Is equivariance exact by construction or encouraged by data augmentation?
- Does preprocessing introduce a frame that changes the symmetry assumption?
- Are vector or coordinate targets transformed consistently during augmentation?
- Is the same equivariance contract valid for train, validation, test, and deployment inputs?
- Does the [[concepts/geometric-deep-learning/coordinate-modeling-contract|coordinate modeling contract]] state output type, loss, and metric?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
