---
title: Tensor Field Network
tags:
  - geometric-deep-learning
  - equivariance
  - tensor-field-network
---

# Tensor Field Network

A tensor field network is an equivariant neural architecture that represents features by their transformation type under 3D rotations.

## Why It Matters

- It is an important design pattern for handling scalar, vector, and higher-order geometric features.
- The architecture connects neural message passing with [[concepts/geometric-deep-learning/spherical-harmonics|spherical harmonics]] and representation theory.
- It helps frame why equivariant models need structured feature channels instead of only scalar embeddings.

## Checks

- Which tensor orders are used, and are they justified by the task?
- Are message passing operations equivariant at every layer?
- Is the added expressivity worth the compute and implementation complexity?
- Are final outputs converted to invariant or equivariant quantities correctly?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
