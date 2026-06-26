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

## Site Definition Types

Binding sites can be defined in several ways:

- Ligand-defined: residues near a co-crystallized ligand.
- Geometry-defined: cavities or pockets detected from protein shape.
- Homology-transferred: sites inferred from related structures.
- Motif-defined: residues matching known sequence or structural motifs.
- Docking-box-defined: a user-provided search region.
- Model-predicted: output of a pocket detection or site prediction model.

These are not interchangeable. A ligand-defined site can be unavailable at deployment time, while a predicted site adds another source of uncertainty.

## Atom-Level View

At atom resolution, the site can be represented as protein atoms within a cutoff:

$$
\mathcal{A}_{\mathrm{site}}
=
\{a_i : \min_j \lVert p_i-l_j\rVert_2 < \tau\}
$$

where $p_i$ is a protein atom coordinate and $l_j$ is a ligand atom coordinate. Residue-level and atom-level cutoffs can produce different site boundaries.

## Deployability

A binding-site definition should be available under the intended use case:

$$
\operatorname{site}_{\mathrm{train}}
\approx
\operatorname{site}_{\mathrm{deploy}}
$$

If training uses a bound ligand to define the site but deployment uses an apo protein and a predicted pocket, the model is exposed to a distribution shift.

## Leakage Risk

Ligand-defined pockets can leak answer information. If the pocket is centered or oriented by the known bound ligand, the model may learn where the ligand was rather than how to find or score a site:

$$
\text{bound ligand}
\rightarrow
\text{site frame}
\rightarrow
\text{model input}
$$

This is acceptable for some pose-refinement tasks, but not for claims about blind docking or pocket discovery.

## Key Ideas

- A binding site can be known from a co-crystal structure, predicted from geometry, inferred from homologs, or defined by a docking box.
- Different definitions produce different pocket residues and different model inputs.
- Binding sites are local but depend on global protein conformation and chain context.
- Missing residues, alternate conformations, cofactors, waters, and protonation states can affect site interpretation.
- Site definitions should be versioned because small cutoff or cleaning changes can alter downstream features.

## Practical Checks

- Is the site experimentally known, predicted, transferred, or manually defined?
- Which residues, cofactors, waters, and chains are included?
- Is the binding site definition available at deployment time?
- Does the split prevent the same or similar binding site from leaking across train and test?
- Are pocket-level and full-protein context evaluated separately?
- Is the pocket centered or oriented using information that would be unavailable at inference?
- Are apo/holo conformational changes and missing residues documented?
- Are waters, metal ions, cofactors, and alternate locations handled consistently?

## Related

- [[entities/pocket|Pocket]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
