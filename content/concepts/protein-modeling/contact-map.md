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

## Checks

- Which atom represents a residue: $C_\alpha$, $C_\beta$, backbone center, or all-atom minimum?
- What cutoff defines contact?
- Are sequence-neighbor contacts separated from long-range contacts?
- Does the map ignore chirality, orientation, or side-chain geometry needed by the task?

## Related

- [[entities/structure|Structure]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
