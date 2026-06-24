---
title: Structure
tags:
  - entities
  - structure
  - geometry
---

# Structure

A structure is a spatial arrangement of atoms, residues, or coarse-grained sites with coordinates and geometry.

## Why It Matters

- Structure carries the geometric information needed for docking, folding, and interaction modeling.
- [[concepts/geometric-deep-learning/equivariance|Equivariance]] helps models treat rotated or translated structures consistently.
- Structural representations must distinguish coordinates, frames, distances, and chemical identity.

## Checks

- What coordinate frame, atom subset, and resolution are used?
- Are missing atoms, alternate conformations, and flexible regions handled?
- Does the model use distances only, coordinates, local frames, or full equivariant features?
- Is the output invariant, equivariant, or a mixture of scalar and coordinate predictions?

## Related

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
