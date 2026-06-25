---
title: Structure
tags:
  - entities
  - structure
  - geometry
---

# Structure

A structure is a spatial arrangement of atoms, residues, or coarse-grained sites with coordinates and geometry.

A structure can be written as typed coordinates:

$$
S = \{(a_i, x_i)\}_{i=1}^{N},
\qquad
x_i \in \mathbb{R}^3
$$

where $a_i$ is atom, residue, or site identity and $x_i$ is its coordinate.

## Why It Matters

- Structure carries the geometric information needed for docking, folding, and interaction modeling.
- [[concepts/geometric-deep-learning/equivariance|Equivariance]] helps models treat rotated or translated structures consistently.
- Structural representations must distinguish coordinates, frames, distances, and chemical identity.
- A structure can be represented as coordinates, distances, contact maps, residue graphs, surfaces, grids, or local frames.

## Checks

- What coordinate frame, atom subset, and resolution are used?
- Are missing atoms, alternate conformations, and flexible regions handled?
- Does the model use distances only, coordinates, local frames, or full equivariant features?
- Is the output invariant, equivariant, or a mixture of scalar and coordinate predictions?
- Is the structure experimental, predicted, relaxed, docked, or generated?
- Does preprocessing leak a reference ligand, template, or future evaluation context?

## Related

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
