---
title: Equivariant GNN
tags:
  - geometric-deep-learning
  - gnn
  - equivariance
---

# Equivariant GNN

An equivariant graph neural network preserves known geometric transformations in its outputs. For molecular modeling, this usually means rotations and translations of coordinates should transform predictions consistently.

For coordinate outputs, SE(3) equivariance requires:

$$
F(RX+t) = RF(X)+t
$$

where $X$ is a set of coordinates, $R$ is a rotation, and $t$ is a translation.

For node coordinates $X\in\mathbb{R}^{N\times 3}$, write the transform as:

$$
X' = XR^\top + \mathbf{1}t^\top
$$

An equivariant GNN layer can combine invariant scalar messages with equivariant vector updates:

$$
m_{ij}^{(l)}
=
\phi_m(h_i^{(l)},h_j^{(l)},\lVert x_i-x_j\rVert_2,e_{ij})
$$

$$
h_i^{(l+1)}
=
\phi_h\left(h_i^{(l)},\sum_{j\in\mathcal{N}(i)}m_{ij}^{(l)}\right)
$$

$$
x_i^{(l+1)}
=
x_i^{(l)}
+
\sum_{j\in\mathcal{N}(i)}
\phi_x(m_{ij}^{(l)})(x_i^{(l)}-x_j^{(l)})
$$

The scalar message stays invariant while the coordinate delta rotates with the input.

## Role in Molecular Modeling

- Encode atoms, residues, and interactions as graphs.
- Respect 3D geometry in protein and ligand structures.
- Support coordinate prediction, pose refinement, and structure-aware ranking.

## Invariant vs Equivariant Readouts

- Scalar readout: pool invariant node features for affinity, energy, class, or ranking.
- Vector readout: predict forces, score fields, or velocity fields.
- Coordinate readout: refine poses, conformers, backbone coordinates, or generated structures.
- Pair readout: predict distances, contacts, interactions, or edge labels.

## Failure Modes

- Graph edges are built from future or label-dependent geometry.
- Protein and ligand graphs leak the known bound pose into a screening setting.
- Atom or residue order changes without permutation-safe aggregation.
- Chirality or stereochemistry is lost in graph features while the task depends on it.
- Reported performance comes from random splits rather than scaffold, protein-family, or complex-level splits.

## Design Checks

- Which quantities are invariant and which are equivariant?
- Are edge features local, global, or both?
- Does the model preserve the symmetries needed by [[concepts/generative-models/flow-matching|flow matching]] or docking?
- Are node and edge updates using relative geometry rather than absolute coordinates?
- Does the readout match the target: scalar score, vector field, coordinates, or pose?
- Is graph construction fit only from inputs available at inference time?
- Are stereochemistry, protonation, conformer source, and coordinate units specified?

## Related

- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
