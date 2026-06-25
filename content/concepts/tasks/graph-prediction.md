---
title: Graph Prediction
tags:
  - tasks
  - graph
  - structured-prediction
---

# Graph Prediction

Graph prediction outputs a graph, a graph property, node labels, edge labels, or a graph edit. It is a task family for molecules, knowledge graphs, interaction networks, scene graphs, dependency graphs, and program structures.

A graph is:

$$
G=(V,E,X,A)
$$

where $V$ is the node set, $E$ is the edge set, $X$ contains node or edge features, and $A$ is an adjacency representation.

## Output Types

- Graph-level prediction: class, scalar, or ranking score for a whole graph.
- Node prediction: label or value for each node.
- Edge prediction: link existence, relation type, distance, bond type, or contact.
- Graph generation: produce nodes, edges, labels, and attributes.
- Graph edit prediction: add, remove, or relabel nodes and edges.

For edge prediction:

$$
\hat{A}_{ij}
=
\sigma(s_\theta(h_i,h_j,e_{ij}))
$$

where $h_i$ and $h_j$ are node representations and $e_{ij}$ is optional pair evidence.

For graph generation:

$$
p_\theta(G)
=
p_\theta(V)
p_\theta(E\mid V)
p_\theta(X,A\mid V,E)
$$

This factorization is schematic; the concrete decoder must define a valid ordering, constraint set, and stopping rule.

## Evaluation Risks

- Node ordering should not change the meaning of the output.
- Near-duplicate graphs can leak across train/test splits.
- Graph-level labels can hide node-level or edge-level failure.
- Generated graphs may be syntactically valid but chemically, physically, or logically invalid.
- Negative edges may be unobserved rather than true negatives.

## Checks

- Is the task graph-level, node-level, edge-level, generative, or edit-based?
- What makes a graph valid?
- Is the graph directed, undirected, typed, weighted, dynamic, or geometric?
- Are node and edge identities aligned across examples?
- Does evaluation respect permutation invariance?
- What split unit prevents graph overlap or near-duplicate leakage?

## Related

- [[concepts/modalities/graph|Graph]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/tasks/interaction-prediction|Interaction prediction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/evaluation/leakage|Leakage]]
