---
title: Semi-Supervised Classification with Graph Convolutional Networks
aliases:
  - papers/gcn
  - papers/graph-convolutional-networks
tags:
  - papers
  - architectures
  - graph-neural-network
---

# Semi-Supervised Classification with Graph Convolutional Networks

> The paper popularized a simple layer-wise graph convolutional network for semi-supervised node classification on graph-structured data.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Semi-Supervised Classification with Graph Convolutional Networks |
| Authors | Thomas N. Kipf, Max Welling |
| Year | 2016 preprint; 2017 conference |
| Venue | ICLR 2017 |
| arXiv | [1609.02907](https://arxiv.org/abs/1609.02907) |
| OpenReview | [SJU4ayYgl](https://openreview.net/forum?id=SJU4ayYgl) |
| Status | verified |

## Question

Graphs do not have a regular grid like images or a simple sequence order like text. The question was how to define a neural layer that mixes node features through graph structure while remaining simple enough for semi-supervised learning.

## Main Claim

A first-order approximation to spectral graph convolution yields an efficient neural layer for graph-based semi-supervised node classification.

Narrowed claim:

$$
H^{(\ell+1)}
=
\sigma
\left(
\tilde{D}^{-1/2}
\tilde{A}
\tilde{D}^{-1/2}
H^{(\ell)}
W^{(\ell)}
\right)
$$

where $\tilde{A}=A+I$ adds self-loops and $\tilde{D}$ is its degree matrix.

## Method

The GCN layer has two coupled operations:

| Operation | Role |
| --- | --- |
| feature projection | apply learnable weights to node features |
| normalized neighbor aggregation | mix each node with its graph neighbors |

The model applies the supervised loss only on labeled nodes:

$$
\mathcal{L}
=
-\sum_{i\in \mathcal{Y}_L}
\sum_{c=1}^{C}
Y_{ic}\log Z_{ic}
$$

This makes the graph structure propagate information from labeled to unlabeled nodes.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| GCN works for semi-supervised node classification | citation-network benchmarks | transductive graph setting is central |
| Normalized graph propagation is a useful architecture primitive | comparisons to prior graph-based methods | benchmark scale is modest |
| Graph convolution is efficient and simple | sparse matrix operations over graph edges | performance depends on graph quality |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | semi-supervised node classification |
| Input/output unit | graph with node features to node labels |
| Main datasets | citation networks and related graph benchmarks |
| Main metric | classification accuracy |
| Not directly tested | large molecular docking graphs, dynamic graphs, graph generation |

## Limitations

- GCN assumes the graph edges are meaningful for the label propagation problem.
- Stacking many layers can cause over-smoothing unless the architecture is modified.
- The canonical benchmarks are small compared with modern graph learning workloads.
- Homophily assumptions may fail in heterophilous graphs.

## Why It Matters

GCN is a canonical starting point for graph neural networks because it makes graph message passing concrete and easy to compare.

## Connections

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[papers/architectures/index|Architecture papers]]
