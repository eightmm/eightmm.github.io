---
title: Graph Construction
tags:
  - architectures
  - graphs
  - molecular-modeling
---

# Graph Construction

Graph construction decides which nodes and edges a model sees. For molecular and protein modeling, it is often as important as the GNN architecture itself.

A radius graph connects nodes within a cutoff:

$$
(i,j)\in E
\quad\Longleftrightarrow\quad
\lVert x_i-x_j\rVert_2 \le r
$$

A k-nearest neighbor graph connects each node to its nearest neighbors:

$$
\mathcal{N}(i)
= \operatorname{kNN}(x_i;\{x_j\}_{j\ne i}, k)
$$

## Common Choices

| Graph Type | Nodes | Edges | Typical Use |
| --- | --- | --- | --- |
| [Chemical bond graph](/concepts/molecular-modeling/molecular-graph) | atoms | covalent bonds | molecular property prediction, graph pretraining |
| contact map | residues | distance/contact threshold | protein structure and function modeling |
| radius graph | atoms, residues, points | $\|x_i-x_j\|_2 \le r$ | local 3D message passing |
| k-nearest neighbor graph | atoms, residues, embeddings | nearest neighbors by distance or similarity | fixed-degree geometric models |
| bipartite interaction graph | protein nodes and ligand nodes | cross edges between sets | protein-ligand interaction, docking, scoring |
| fully connected graph | all nodes | all pairs | small molecules, attention-heavy graph transformers |

## Construction Function

Write graph construction as a function:

$$
G
=
C_\psi(u, r, c)
=
(V,E,A)
$$

where $u$ is the raw entity, $r$ is the representation, $c$ is context such as pocket or conformer protocol, and $A$ contains node, edge, or pair attributes. This makes clear whether graph construction is deterministic, learned, cached, or context-dependent.

## Edge Attributes

| Attribute | Meaning | Risk |
| --- | --- | --- |
| bond type | chemical relation | aromaticity, charge, stereo, and hydrogens depend on standardization |
| distance | coordinate relation | coordinate source and unit must be fixed |
| direction vector | equivariant relation | frame and rotation behavior must be explicit |
| sequence separation | residue relation | chain breaks and missing residues matter |
| interaction type | protein-ligand relation | may be derived from known pose or post-hoc annotation |
| learned relation | model-discovered edge | can hide data leakage if fit using test data |

## Leakage Boundary

Graph construction can leak when edges use information unavailable at inference:

$$
E_{\mathrm{eval}}
\not\subseteq
E_{\mathrm{inference}}
\Rightarrow
\text{claim mismatch}
$$

Examples include ligand-defined pockets in blind docking, contact maps from target structures unavailable in deployment, similarity graphs built using the full benchmark, or conformer/pose edges generated after seeing labels.

## Checks

- What is a node: atom, residue, conformer, pocket point, or abstract entity?
- Are edges chemical, spatial, sequential, learned, or task-specific?
- Does graph construction use target information or future data?
- Are cutoff and $k$ values stable across molecule and protein sizes?
- Is graph construction performed before or after the split?
- Are coordinates, conformers, pockets, and templates available at inference?
- Are node ordering, atom mapping, and residue indexing stable across preprocessing?

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
