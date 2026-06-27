---
title: Discrete Math and Graphs
tags:
  - math
  - graphs
  - discrete-math
---

# Discrete Math and Graphs

Discrete math is the language for tokens, sets, graphs, trees, masks, indices, neighborhoods, retrieval candidates, and search spaces. It is especially important for [[concepts/architectures/gnn|Graph neural networks]], molecular graphs, protein contact maps, and agent workflows.

## Core Notes

- [[concepts/modalities/graph|Graph]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/protein-modeling/contact-map|Contact map]]

## Graph Objects

A graph is often written as:

$$
G=(V,E)
$$

where $V$ is a set of nodes and $E$ is a set of edges. With node features and edge features:

$$
X \in \mathbb{R}^{|V|\times d_v},
\quad
E_{ij} \in \mathbb{R}^{d_e}
$$

The adjacency matrix records connectivity:

$$
A_{ij}
=
\begin{cases}
1 & (i,j)\in E \\
0 & \text{otherwise}
\end{cases}
$$

## Neighborhoods

Message passing usually aggregates over a node neighborhood:

$$
\mathcal{N}(i)
=
\{j \mid (j,i)\in E\}
$$

A generic update is:

$$
h_i^{(t+1)}
=
\phi
\left(
h_i^{(t)},
\operatorname{AGG}_{j\in\mathcal{N}(i)}
\psi(h_i^{(t)},h_j^{(t)},e_{ij})
\right)
$$

The aggregation must usually be permutation-invariant over neighbor order.

## Sets and Permutations

For a set function $f(\{x_1,\ldots,x_n\})$, input order should not matter:

$$
f(x_1,\ldots,x_n)
=
f(x_{\pi(1)},\ldots,x_{\pi(n)})
$$

for any permutation $\pi$. This is the core constraint behind set models, pooling, readout, and many graph aggregators.

## Paths and Connectivity

Shortest paths, connected components, graph distance, and neighborhood radius define what information can reach a node:

$$
\operatorname{dist}_G(i,j)
=
\text{length of the shortest path from } i \text{ to } j
$$

In an $L$-layer message passing GNN, node $i$ can usually receive information from at most its $L$-hop neighborhood.

## Computational Biology Connections

- Molecular structure: atoms as nodes, bonds or spatial contacts as edges.
- Protein modeling: residues as nodes, contact maps or distance cutoffs as edges.
- Structure-based modeling: ligand, pocket, and interaction graphs.
- Retrieval: candidates, neighborhoods, ranking sets, and graph-based indexes.

## Checks

- What is the node unit: atom, residue, token, document, tool, or state?
- Are edges chemical bonds, spatial contacts, sequence adjacency, learned attention, or retrieval links?
- Is the graph directed, undirected, weighted, dynamic, or heterogeneous?
- Which operations must be permutation-invariant or permutation-equivariant?
- Does the model need local neighborhoods, global attention, or both?

## Related

- [[math/index|Math]]
- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/protein-modeling/contact-map|Contact map]]
