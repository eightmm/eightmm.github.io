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

Contact maps are invariant to global rigid motion:

$$
D_{ij}(RX+t)=D_{ij}(X)
$$

for rotation $R$ and translation $t$. They discard global orientation and much of local frame information.

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

## Map Types

| Map | Formula / Value | Use |
| --- | --- | --- |
| binary contact | $\mathbf{1}[D_{ij}\le\tau]$ | graph edge, classification target |
| distance map | $D_{ij}$ | geometry supervision and comparison |
| binned distance | class over distance intervals | structure prediction target |
| orientation-aware map | distance plus angles/frames | richer fold constraints |
| inter-chain contact | residue pairs across chains | complex/interface modeling |
| pocket-ligand contact | residue/atom vs ligand atom | SBDD interaction features |

The map type should match the downstream claim. A binary contact map cannot fully recover chirality, side-chain orientation, or atom-level clashes.

## Loss and Metrics

For binary contact prediction, a common loss is cross-entropy over residue pairs:

$$
\mathcal{L}
=
-\sum_{i<j}
\left[
C_{ij}\log \hat{p}_{ij}
+
(1-C_{ij})\log(1-\hat{p}_{ij})
\right]
$$

Because contacts are sparse, evaluation often uses top-$L$, top-$L/2$, precision, or long-range contact precision rather than plain accuracy.

For distance maps, regression or binned classification can be used:

$$
\mathcal{L}_{D}
=
\sum_{i<j}
\rho(\hat{D}_{ij},D_{ij})
$$

where $\rho$ should respect missing residues and unresolved regions.

## Use in Models

Contact maps can be:

- prediction targets
- graph edges
- structure comparison summaries
- constraints for folding or refinement
- diagnostic views for sequence-structure models

Before building a contact map from coordinates, verify [[concepts/protein-modeling/residue-indexing|Residue indexing]] and [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]].

## Leakage and Preprocessing

Contact maps derived from experimental or predicted structures can leak template information:

| Source | Risk |
| --- | --- |
| experimental structure | may overlap with benchmark template or homolog |
| predicted structure | teacher-model leakage if used as label or input |
| MSA-derived contacts | family-level information can cross splits |
| ligand-defined pocket contacts | unavailable in blind deployment |
| missing residues treated as non-contact | false negatives from unresolved structure |

## Checks

- Which atom represents a residue: $C_\alpha$, $C_\beta$, backbone center, or all-atom minimum?
- What cutoff defines contact?
- Are sequence-neighbor contacts separated from long-range contacts?
- Does the map ignore chirality, orientation, or side-chain geometry needed by the task?
- Are missing residues masked rather than treated as far-away residues?
- Is the contact definition identical across train, validation, test, and inference?
- Is the contact map an input, target, metric, graph edge rule, or diagnostic?
- Are local, medium-range, long-range, and inter-chain contacts separated when needed?
- Does the split prevent homolog/template leakage?

## Related

- [[entities/structure|Structure]]
- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
