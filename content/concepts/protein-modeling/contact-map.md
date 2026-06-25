---
title: Contact Map
tags:
  - protein-modeling
  - structure
---

# Contact Map

A contact map records which residue pairs are close in 3D space. It is a compact structural representation used for folding, structure comparison, and graph construction.

Given residue coordinates $x_i$, a binary contact map is:

$$
C_{ij}
= \mathbf{1}
\left[
\lVert x_i-x_j\rVert_2 \le \tau
\right]
$$

where $\tau$ is a distance cutoff.

A soft distance map keeps the continuous geometry:

$$
D_{ij}
= \lVert x_i-x_j\rVert_2
$$

## Contact Definition

The definition of $x_i$ changes the map:

- $C_\alpha$: simple backbone-based distance.
- $C_\beta$: common for residue contact maps, with glycine handled specially.
- Minimum heavy-atom distance: side-chain-aware but sensitive to atom completeness.
- Residue center: coarse but stable for graph construction.

Contacts can also exclude local sequence neighbors:

$$
|i-j| > k
$$

so that long-range structural contacts are not dominated by adjacent residues.

## Use in Models

Contact maps can be:

- prediction targets
- graph edges
- structure comparison summaries
- constraints for folding or refinement
- diagnostic views for sequence-structure models

Before building a contact map from coordinates, verify [[concepts/protein-modeling/residue-indexing|Residue indexing]] and [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]].

## Checks

- Which atom represents a residue: $C_\alpha$, $C_\beta$, backbone center, or all-atom minimum?
- What cutoff defines contact?
- Are sequence-neighbor contacts separated from long-range contacts?
- Does the map ignore chirality, orientation, or side-chain geometry needed by the task?
- Are missing residues masked rather than treated as far-away residues?
- Is the contact definition identical across train, validation, test, and inference?

## Related

- [[entities/structure|Structure]]
- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
