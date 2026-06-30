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

## Layer Contract

An equivariant GNN layer should state:

| Component | Expected Behavior |
| --- | --- |
| scalar node features $h_i$ | invariant to global rigid motion |
| coordinates $x_i$ | transform by rotation and translation |
| relative vector $x_i-x_j$ | rotates, does not translate |
| distance $\|x_i-x_j\|$ | invariant |
| vector output | equivariant |
| scalar readout | invariant |

The architecture is only as equivariant as the features and preprocessing supplied to it.

## Common Variants

| Variant | Main Idea | Caveat |
| --- | --- | --- |
| scalar-distance message passing | use invariant distances and scalar messages | may lose orientation or chirality |
| coordinate-update networks | move points using relative vectors | stability and unit scale matter |
| tensor/irreducible representation models | carry higher-order geometric features | more complex and expensive |
| equivariant attention | attention over scalar and geometric features | masking, frames, and pair features define symmetry |
| local-frame models | represent geometry in residue or pocket frames | frame construction can leak or break equivariance |

## Representative Paper Notes

| Note | Why it matters |
| --- | --- |
| [[papers/architectures/tensor-field-networks|Tensor Field Networks]] | establishes typed tensor features and spherical-harmonic filters for 3D equivariant layers |
| [[papers/architectures/se3-transformer|SE(3)-Transformer]] | adds attention-style weighting to equivariant 3D representations |
| [[papers/architectures/egnn|EGNN]] | gives a simpler coordinate-update view of equivariant graph networks |
| [[papers/architectures/painn|PaiNN]] | uses scalar and vector atom features for molecular tensorial properties |
| [[papers/architectures/nequip|NequIP]] | applies E(3)-equivariant tensor features to interatomic potentials and force learning |

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
- The model is called equivariant but only the coordinate update, not the full pipeline, is equivariant.
- Reflection behavior is mismatched with chirality-sensitive molecular tasks.
- Evaluation aligns outputs in a way that hides frame or permutation errors.

## Design Checks

- Which quantities are invariant and which are equivariant?
- Are edge features local, global, or both?
- Does the model preserve the symmetries needed by [[concepts/generative-models/flow-matching|flow matching]] or docking?
- Are node and edge updates using relative geometry rather than absolute coordinates?
- Does the readout match the target: scalar score, vector field, coordinates, or pose?
- Is graph construction fit only from inputs available at inference time?
- Are stereochemistry, protonation, conformer source, and coordinate units specified?
- Is equivariance tested by applying random rigid transforms to the same input?
- Are atom/residue mappings and symmetry corrections fixed before evaluation?

## Related

- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
