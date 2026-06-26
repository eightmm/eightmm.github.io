---
title: Graph Transformers
tags:
  - architectures
  - graph-transformer
  - graphs
  - attention
---

# Graph Transformers

Graph transformers combine graph-structured inputs with attention-based message mixing. They are a bridge between [[concepts/architectures/gnn|GNNs]] and [[concepts/architectures/transformer|Transformers]].

A simple graph-biased attention score is:

$$
a_{ij}
= \frac{q_i^\top k_j}{\sqrt{d}}
+ b(e_{ij})
$$

where $b(e_{ij})$ injects edge, distance, or relation information into attention.

For multi-head graph attention:

$$
\operatorname{head}_h(i)
=
\sum_{j\in \mathcal{C}(i)}
\operatorname{softmax}_{j}
\left(
\frac{q_{i,h}^{\top}k_{j,h}}{\sqrt{d_h}}
+ b_h(i,j)
\right)
v_{j,h}
$$

where $\mathcal{C}(i)$ can be all nodes, a neighborhood, or a sparse candidate set. The choice of $\mathcal{C}(i)$ determines both inductive bias and cost.

## Key Ideas

- Attention can connect distant graph nodes without many local message-passing steps.
- Structural encodings tell the model about edges, distances, paths, centrality, or geometry.
- Some variants attend over all node pairs; others restrict attention to neighborhoods or sparse patterns.
- Edge or pair representations often sit beside node representations and carry relation-specific information.
- Geometry-aware graph transformers connect to [[concepts/geometric-deep-learning/index|geometric deep learning]] when coordinates, frames, or equivariant features are used.

## Structural Biases

| Bias | Encodes | Caveat |
| --- | --- | --- |
| adjacency bias | graph connectivity | graph construction becomes part of the model |
| shortest-path distance | topology | expensive or ambiguous on disconnected graphs |
| pair distance | 3D geometry | coordinate source and unit matter |
| edge type | bond, contact, interaction, relation | edge labels may leak post-hoc information |
| centrality or degree | coarse topology | can overfit graph-size artifacts |
| relative position | sequence or spatial offset | frame and masking rules must be clear |

## Cost Boundary

Full node-pair attention is:

$$
O(N^2 d)
$$

for $N$ nodes and hidden size $d$. For proteins, complexes, and large graphs, the paper should state whether attention is full, local, sparse, blockwise, or biased by a candidate graph.

## Claim Boundary

| Claim | Check |
| --- | --- |
| better long-range reasoning | compare against message-passing depth and long-range baselines |
| better geometric modeling | state coordinate features, symmetry behavior, and frame assumptions |
| better molecular/protein performance | separate architecture gain from graph construction and split choice |
| better efficiency | compare memory, wall time, and node/pair scaling |

## Practical Checks

- Identify whether graph edges only bias attention or also carry explicit messages.
- Check how positional, distance, or relation encodings enter the attention score and value update.
- Watch quadratic cost in atom, residue, or contact count.
- For protein-ligand complexes, check whether intra-protein, intra-ligand, and cross interactions are separated or shared.
- Check whether edge or pair features are updated, only used as bias, or discarded after attention.
- Verify that graph construction and structural encodings are available at inference.

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
