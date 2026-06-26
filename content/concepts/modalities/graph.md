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

## Graph Construction Boundary

A graph is often derived from another object:

$$
G
=
\operatorname{construct\_graph}(x;\ \rho)
$$

where $\rho$ contains rules such as distance cutoffs, k-nearest neighbors, bond definitions, contact thresholds, relation extraction, or temporal adjacency. Changing $\rho$ changes the task input.

## Representation Contract

A graph input should state:

$$
\mathcal{G}
=
(V,E,X,E_f,M)
$$

where $V$ is the node set, $E$ is the edge set, $X$ are node features, $E_f$ are edge features, and $M$ stores masks for valid nodes, edges, or subgraphs.

| Domain | Node | Edge | Split Risk |
| --- | --- | --- | --- |
| molecule | atom | bond, distance, contact | same scaffold or standardized molecule across splits |
| protein | residue, atom, domain | contact, sequence adjacency, spatial neighbor | homolog or template leakage |
| protein-ligand complex | atom, residue, ligand fragment | bond, contact, distance, interaction | same protein family or ligand scaffold |
| knowledge graph | entity | relation triple | entity leakage across train/test edges |
| agent trace | state, message, tool call | transition, dependency, citation | same task template or hidden state reuse |

## Permutation Contract

If node order is arbitrary, a graph-level output should be invariant:

$$
f(PX, PAP^\top)=f(X,A)
$$

where $P$ is a permutation matrix. Node-level outputs should be equivariant:

$$
F(PX, PAP^\top)=P F(X,A)
$$

This distinction should be stated before choosing pooling, attention, or message passing.

## Output Levels

Graph tasks differ by prediction unit:

$$
Y
\in
\{\text{node labels},
\text{edge labels},
\text{graph label},
\text{subgraph},
\text{path},
\text{generated graph}\}
$$

The pooling/readout, loss, metric, and split should match the output level. A graph-level split does not necessarily prevent node or entity leakage.

## Edge Semantics

Edges can mean very different things:

| Edge Type | Meaning | Risk |
| --- | --- | --- |
| observed bond/relation | part of the raw object | may miss alternative chemical states or relation uncertainty |
| distance cutoff | geometric proximity | cutoff is a modeling hyperparameter |
| k-nearest neighbor | fixed local density | can connect through unavailable deployment context |
| learned edge | inferred during model execution | must not use test labels or future information |
| temporal edge | event order | future edges can leak labels |

For molecular and protein graphs, edge construction is often as important as the GNN architecture.

## Leakage Risks

- The same entity appears in train and test through different graph neighborhoods.
- Edges are constructed using labels or future information.
- Global normalization or message passing crosses examples during batching.
- Random row splits leak near-duplicate molecular graphs, protein contact maps, or knowledge-graph entities.

## Practical Checks

- What are nodes and edges?
- Are edges observed, constructed, thresholded, or learned?
- Are node and edge features available at deployment?
- Does the split prevent near-duplicate graphs or entity leakage?
- Is the task node-level, edge-level, graph-level, or subgraph-level?
- Does batching preserve graph isolation?
- Are edge construction rules fixed before evaluation?
- Is the model expected to be permutation invariant or equivariant?
- Does the edge definition preserve the task's physical, biological, or relational meaning?
- Are graph-level metrics hiding node-level or edge-level failure modes?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/tasks/graph-prediction|Graph prediction]]
- [[concepts/tasks/interaction-prediction|Interaction prediction]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/evaluation/leakage|Leakage]]
