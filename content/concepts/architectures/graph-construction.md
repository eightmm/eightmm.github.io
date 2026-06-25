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

- [[concepts/molecular-modeling/molecular-graph|Chemical bond graph]] for molecules.
- Contact map for proteins.
- Radius graph over 3D coordinates.
- k-nearest neighbor graph over coordinates or learned embeddings.
- Bipartite protein-ligand graph for interaction modeling.

## Checks

- What is a node: atom, residue, conformer, pocket point, or abstract entity?
- Are edges chemical, spatial, sequential, learned, or task-specific?
- Does graph construction use target information or future data?
- Are cutoff and $k$ values stable across molecule and protein sizes?

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
