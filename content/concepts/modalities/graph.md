---
title: Graph
tags:
  - modalities
  - graph
  - structured-data
---

# Graph

A graph modality represents entities and relationships as nodes and edges. Graphs appear in molecules, protein contact maps, knowledge graphs, citation networks, routes, tool-call traces, and interaction networks.

A graph is:

$$
G=(V,E)
$$

with node features $X\in\mathbb{R}^{|V|\times d_v}$ and optional edge features:

$$
e_{ij}\in\mathbb{R}^{d_e},
\qquad
(i,j)\in E
$$

The adjacency matrix records connectivity:

$$
A_{ij}=1
\quad
\text{if}
\quad
(i,j)\in E
$$

## Key Ideas

- Node order is usually arbitrary, so permutation invariance or equivariance matters.
- Edge construction is part of the data representation, not a neutral preprocessing step.
- Graphs can be directed, undirected, typed, weighted, dynamic, or geometric.
- Dense pairwise relationships may be better treated as attention or set modeling; sparse typed relationships often suit graph models.

## Practical Checks

- What are nodes and edges?
- Are edges observed, constructed, thresholded, or learned?
- Are node and edge features available at deployment?
- Does the split prevent near-duplicate graphs or entity leakage?
- Is the task node-level, edge-level, graph-level, or subgraph-level?

## Related

- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/evaluation/leakage|Leakage]]
