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

## Structure Contract

| Field | Question | Example |
| --- | --- | --- |
| Unit | atom, residue, chain, pocket, complex, surface, grid? | protein backbone, ligand atoms |
| Source | experimental, predicted, template-derived, docked, generated? | X-ray, cryo-EM, predicted structure |
| Coordinate frame | global, receptor-aligned, ligand-centered, local residue frame? | arbitrary PDB frame |
| Resolution | all atom, backbone, C-alpha, coarse-grained, surface points? | $C_\alpha$ trace |
| Chemical state | protonation, tautomer, charge, waters, cofactors, metals? | prepared receptor |
| Missing data | unresolved residues, missing atoms, alternate locations? | loop gaps, side-chain completion |
| Allowed information | what was available at inference time? | no test ligand pose |

For coordinate matrices:

$$
X\in\mathbb{R}^{N\times 3}
$$

the model should state whether atom/residue order is meaningful, arbitrary, or tied to a public indexing scheme.

## Symmetry Boundary

Rigid motions should not change scalar labels:

$$
f(XR^\top+\mathbf{1}t^\top)=f(X)
$$

Coordinate outputs should transform with the input:

$$
F(XR^\top+\mathbf{1}t^\top)=F(X)R^\top+\mathbf{1}t^\top
$$

If a paper uses absolute coordinates without explaining the frame, the structure claim is incomplete.

## Source and Evidence

| Structure Source | Claim Strength | Risk |
| --- | --- | --- |
| experimental structure | strongest for measured geometry | resolution, missing atoms, construct differences |
| predicted structure | useful input or hypothesis | model bias and circular evidence |
| template-derived structure | prior from homolog | template leakage |
| docked structure | generated pose | scoring and pose quality must be separated |
| generated structure | model output | validity and physical plausibility required |

Structure source should be recorded before comparing metrics.

## Evaluation Boundary

| Claim | Metric or Check |
| --- | --- |
| coordinate accuracy | RMSD, aligned error, distance error |
| contact prediction | precision/recall over contact map |
| pose quality | pose RMSD, clash, interaction recovery |
| geometry validity | bond lengths, angles, chirality, sterics |
| functional relevance | assay, binding, or downstream task evidence |

## Checks

- What coordinate frame, atom subset, and resolution are used?
- Are missing atoms, alternate conformations, and flexible regions handled?
- Does the model use distances only, coordinates, local frames, or full equivariant features?
- Is the output invariant, equivariant, or a mixture of scalar and coordinate predictions?
- Is the structure experimental, predicted, relaxed, docked, or generated?
- Does preprocessing leak a reference ligand, template, or future evaluation context?
- Are atom/residue mappings and alignment policies specified for coordinate metrics?
- Are invalid or incomplete structures counted rather than silently removed?

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
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/sbdd/pose-quality|Pose quality]]
