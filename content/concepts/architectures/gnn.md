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

For a graph-level prediction, a readout should be permutation invariant:

$$
h_G = \rho(\{h_i^{(T)}: i\in V\})
$$

For node-level outputs, a GNN is usually expected to be permutation equivariant. If $P$ permutes node order, then:

$$
f(PX, PAP^\top) = P f(X,A)
$$

## Message Passing Contract

| Component | Required Detail |
| --- | --- |
| node unit | atom, residue, token, pocket point, conformer, protein, ligand, or abstract entity |
| edge unit | bond, contact, distance edge, sequence adjacency, interaction edge, learned relation |
| edge attributes | bond type, distance, direction, residue separation, pair embedding, interaction label |
| update target | node state, edge state, graph state, coordinate, vector field, score, or ranking |
| aggregation | sum, mean, max, attention, normalized message, or degree-scaled message |
| readout | node-level, edge-level, graph-level, pair-level, or complex-level output |
| depth | number of message passing layers and effective receptive field |

The output type changes what the GNN claim means:

| Output | Example Task | Required Symmetry |
| --- | --- | --- |
| node label | residue site, atom type, node mask | permutation equivariance |
| edge label | contact, bond, interaction | permutation equivariance over node pairs |
| graph scalar | molecular property, affinity, class | permutation invariance |
| pair score | molecule-target score, protein-ligand interaction | invariance within each entity and pair identity |
| coordinates or vectors | pose, force, velocity, update | geometric equivariance in addition to permutation behavior |

## Key Ideas

- Nodes hold entity features; edges hold relationships such as bonds, distances, contacts, or interaction types.
- Message passing repeatedly updates node or edge states from local neighborhoods.
- Readout functions pool graph information for graph-level tasks such as property prediction.
- [[concepts/architectures/graph-construction|Graph construction]] is a modeling choice: chemical bonds, k-nearest neighbors, radius graphs, contact maps, or learned edges change the problem.
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]] gives a reference point for the expressive power of standard message-passing GNNs.
- Geometry-aware variants connect message passing to [[concepts/geometric-deep-learning/equivariance|equivariance]] and coordinate updates.

## Common Failure Modes

| Failure | Symptom | Check |
| --- | --- | --- |
| oversmoothing | node embeddings become too similar after many layers | layer depth, residuals, normalization, pairwise embedding variance |
| oversquashing | long-range information is compressed through narrow neighborhoods | graph diameter, bottleneck edges, attention/global edges |
| graph leakage | edges use labels, future data, full benchmark, known pose, or test statistics | graph construction provenance |
| readout mismatch | graph-level pooling hides rare active sites or interaction regions | task-specific pooling and attention audit |
| split mismatch | train/test share scaffold, homolog, complex pair, or graph-derived artifact | split unit and duplicate controls |
| geometry mismatch | scalar GNN is used for coordinate/vector target without equivariance | target transformation rule |

## Paper Reading Pattern

| Claim | First Evidence |
| --- | --- |
| better GNN architecture | same graph construction, features, split, optimizer budget, and readout |
| better graph representation | ablation over nodes, edges, pair features, and construction rule |
| better long-range reasoning | tasks requiring nonlocal signal, not only more parameters |
| better molecular performance | scaffold/source split and feature baseline |
| better protein/complex performance | protein-family or complex split plus leakage audit |

## Practical Checks

- Check what the graph nodes represent: atoms, residues, ligands, pockets, conformers, or abstract states.
- Check whether edge features include distances, directions, bond orders, residue separation, or learned relation types.
- Watch oversmoothing when many message-passing layers make node states too similar.
- For molecular tasks, inspect split strategy for leakage across scaffolds, protein families, or near-duplicate complexes.
- Verify that graph construction is part of the method or held fixed across baselines.
- Match readout to the output unit before comparing metrics.

## Related

- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/architectures/graph-isomorphism-network|How Powerful are Graph Neural Networks?]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
