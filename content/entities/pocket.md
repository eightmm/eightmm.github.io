---
title: Pocket
tags:
  - entities
  - protein
  - structure-based-ai
---

# Pocket

A pocket is a local region of a protein that can host a ligand, cofactor, substrate, or designed molecule. In structure-based AI, the pocket is often the practical input object rather than the full protein.

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

## Checks

- Is the pocket defined from a known ligand, predicted binding site, or whole-protein scan?
- Does the definition include backbone atoms, side chains, waters, metals, or cofactors?
- Does pocket extraction leak test-set ligand information?
- Is the pocket rigid, flexible, or updated during modeling?

## Related

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
