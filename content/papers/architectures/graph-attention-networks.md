---
title: Graph Attention Networks
aliases:
  - papers/gat
  - papers/graph-attention-networks
tags:
  - papers
  - architectures
  - graph-neural-network
  - attention
---

# Graph Attention Networks

> The paper introduced masked self-attention over graph neighborhoods as a graph neural network layer.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Graph Attention Networks |
| Authors | Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio |
| Year | 2017 preprint; 2018 conference |
| Venue | ICLR 2018 |
| arXiv | [1710.10903](https://arxiv.org/abs/1710.10903) |
| OpenReview | [rJXMpikCZ](https://openreview.net/forum?id=rJXMpikCZ) |
| Status | verified |

## Question

GCN-style layers aggregate neighbors with weights determined mostly by graph structure and normalization. The question was whether each node could learn which neighbors matter more, while keeping computation local to graph neighborhoods.

## Main Claim

Masked self-attention over a node's neighborhood can replace fixed graph convolution weights with learned, feature-dependent neighbor weights.

Narrowed claim:

$$
h_i'
=
\sigma
\left(
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij} W h_j
\right)
$$

where $\alpha_{ij}$ is an attention coefficient normalized over neighbors of node $i$.

## Method

The layer computes an attention logit for each edge:

$$
e_{ij}
=
a(Wh_i, Wh_j)
$$

Then it normalizes over the neighborhood:

$$
\alpha_{ij}
=
\frac{
\exp(\operatorname{LeakyReLU}(e_{ij}))
}{
\sum_{k \in \mathcal{N}(i)}
\exp(\operatorname{LeakyReLU}(e_{ik}))
}
$$

Multi-head graph attention uses several independent attention mechanisms and either concatenates or averages their outputs.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Attention over graph neighborhoods improves graph learning | citation network and protein-protein interaction benchmarks | benchmark scale and graph construction are limited |
| GAT handles inductive graph settings | PPI experiment uses unseen graphs at test time | not all graph domains are inductive in the same way |
| Learned neighbor weights are useful | comparisons to GCN-style methods | attention weights are not automatically explanations |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | node classification |
| Input/output unit | graph with node features to node labels |
| Main datasets | Cora, Citeseer, Pubmed, PPI |
| Main comparison | graph convolution and graph neural baselines |
| Not directly tested | molecular docking, 3D equivariant modeling, dynamic graphs |

## Limitations

- Attention is restricted to observed graph neighborhoods, so graph construction still defines the available information.
- Attention coefficients can be unstable or misleading if interpreted as causal explanations.
- Dense high-degree graphs can make neighbor attention expensive.
- Later graph transformers and message-passing variants changed global attention, edge features, positional encodings, and scalability.

## Why It Matters

GAT is a key bridge between [[concepts/architectures/attention|Attention]] and [[concepts/architectures/gnn|Graph neural networks]]. It shows how attention can become a local graph message-passing rule rather than a dense sequence operation.

## Connections

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[papers/architectures/gcn|Semi-Supervised Classification with GCNs]]
- [[papers/architectures/index|Architecture papers]]
