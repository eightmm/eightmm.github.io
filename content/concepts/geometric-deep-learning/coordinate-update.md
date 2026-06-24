---
title: Coordinate Update
tags:
  - geometric-deep-learning
  - coordinates
  - equivariance
---

# Coordinate Update

A coordinate update is a model step that changes point positions while preserving the intended geometric symmetry.

## Why It Matters

- Coordinate updates appear in structure refinement, docking, diffusion, and flow-based generation.
- Updates should be designed so translated or rotated inputs produce translated or rotated outputs.
- Separating scalar features from coordinate movement helps audit equivariance assumptions.

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
