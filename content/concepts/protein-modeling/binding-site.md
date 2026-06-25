---
title: Binding Site
tags:
  - protein-modeling
  - sbdd
  - structure
---

# Binding Site

A binding site is a region of a protein that interacts with a ligand, substrate, cofactor, peptide, nucleic acid, or another protein. In structure-based modeling, the binding site defines the local context for docking, scoring, and interaction prediction.

A simple geometric definition selects residues within a distance cutoff of a ligand:

$$
\mathcal{B}
=
\{r_i : \min_j \lVert x_i - l_j\rVert_2 < \tau\}
$$

where $r_i$ is a residue, $x_i$ is a representative residue coordinate, $l_j$ is a ligand atom coordinate, and $\tau$ is a cutoff.

## Key Ideas

- A binding site can be known from a co-crystal structure, predicted from geometry, inferred from homologs, or defined by a docking box.
- Different definitions produce different pocket residues and different model inputs.
- Binding sites are local but depend on global protein conformation and chain context.
- Missing residues, alternate conformations, cofactors, waters, and protonation states can affect site interpretation.

## Practical Checks

- Is the site experimentally known, predicted, transferred, or manually defined?
- Which residues, cofactors, waters, and chains are included?
- Is the binding site definition available at deployment time?
- Does the split prevent the same or similar binding site from leaking across train and test?
- Are pocket-level and full-protein context evaluated separately?

## Related

- [[entities/pocket|Pocket]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
