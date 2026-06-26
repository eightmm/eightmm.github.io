---
title: Pocket
tags:
  - entities
  - protein
  - structure-based-ai
---

# Pocket

A pocket is a local region of a protein that can host a ligand, cofactor, substrate, or designed molecule. In structure-based modeling, the pocket is often the practical input object rather than the full protein.

## Modeling Views

- Residue set around a known ligand or binding site.
- 3D grid, surface patch, atom graph, residue graph, or point cloud.
- Conditional context for docking, pose generation, and scoring.

Given protein coordinates $X_P$ and a ligand reference position $X_L$, a simple distance-based pocket can be defined as:

$$
\mathcal{P}_r
= \{i \in P \mid \min_{j\in L}\lVert x_i-x_j\rVert_2 \le r\}
$$

where $r$ is a cutoff radius.

## Pocket Source

Pocket definitions differ by available evidence:

$$
\text{pocket source}
\in
\{\text{known ligand},
\text{known site annotation},
\text{structure-only detector},
\text{whole-protein scan},
\text{model-generated region}\}
$$

A ligand-defined pocket is valid for retrospective pose analysis, but can leak answer information if the intended deployment setting must find the binding site without a bound ligand.

## Representation Boundary

The pocket can be represented as:

- residues within a cutoff;
- atoms within a cutoff;
- surface points and normals;
- voxel grid;
- residue or atom graph;
- local coordinate frame;
- sequence window around annotated residues.

Each representation makes different assumptions about rigidity, solvent, protonation, side-chain flexibility, and coordinate-frame availability.

## Leakage Risk

Pocket extraction should be part of the protocol:

$$
\text{model input}
=
\operatorname{extract\_pocket}(P,\ \text{allowed evidence})
$$

If the extraction function uses the test ligand pose, bound ligand center, or benchmark-only annotation that would not be available in screening, the task becomes easier than the claimed deployment setting.

## Checks

- Is the pocket defined from a known ligand, predicted binding site, or whole-protein scan?
- Does the definition include backbone atoms, side chains, waters, metals, or cofactors?
- Does pocket extraction leak test-set ligand information?
- Is the pocket rigid, flexible, or updated during modeling?
- Is the same pocket extraction protocol used for train, validation, test, and inference?
- Are apo and holo structures handled consistently?
- Are local coordinate frames defined without using unavailable ligand information?

## Related

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
