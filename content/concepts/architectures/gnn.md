---
title: Graph Neural Networks
tags:
  - architectures
  - gnn
  - graphs
---

# Graph Neural Networks

Graph neural networks operate on nodes, edges, and message passing. In molecular modeling, atoms, residues, bonds, contacts, and interactions can all be represented as graph structure.

A common message-passing update is:

$$
m_i^{(t)}
= \sum_{j\in \mathcal{N}(i)}
\phi_m(h_i^{(t)}, h_j^{(t)}, e_{ij})
$$

$$
h_i^{(t+1)}
= \phi_u(h_i^{(t)}, m_i^{(t)})
$$

Here $h_i$ is a node state, $e_{ij}$ is an edge feature, and $\mathcal{N}(i)$ is the neighborhood of node $i$.

## Key Ideas

- Nodes hold entity features; edges hold relationships such as bonds, distances, contacts, or interaction types.
- Message passing repeatedly updates node or edge states from local neighborhoods.
- Readout functions pool graph information for graph-level tasks such as property prediction.
- Edge construction is a modeling choice: chemical bonds, k-nearest neighbors, radius graphs, contact maps, or learned edges change the problem.
- Geometry-aware variants connect message passing to [[concepts/geometric-deep-learning/equivariance|equivariance]] and coordinate updates.

## Practical Checks

- Check what the graph nodes represent: atoms, residues, ligands, pockets, conformers, or abstract states.
- Check whether edge features include distances, directions, bond orders, residue separation, or learned relation types.
- Watch oversmoothing when many message-passing layers make node states too similar.
- For molecular tasks, inspect split strategy for leakage across scaffolds, protein families, or near-duplicate complexes.

## Related

- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
