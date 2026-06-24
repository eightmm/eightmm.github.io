---
title: Equivariance
tags:
  - geometric-deep-learning
  - symmetry
  - equivariance
---

# Equivariance

Equivariance means a model output transforms predictably when the input is transformed.

## Why It Matters

- Coordinate predictions should rotate or translate with the input structure.
- Equivariance reduces the burden of learning symmetries from data alone.
- Many geometric neural networks combine equivariant internal features with invariant task outputs.

## Checks

- What transformation group is being considered?
- Which model outputs should transform, and which should remain invariant?
- Is equivariance exact by construction or encouraged by data augmentation?
- Does preprocessing introduce a frame that changes the symmetry assumption?

## Related

- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
