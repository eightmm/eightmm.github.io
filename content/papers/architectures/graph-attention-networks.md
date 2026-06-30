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

The deeper architecture question is how to move attention from dense sequences to sparse graphs. A Transformer attends over all token pairs in a sequence. A graph neural network usually restricts communication to edges. GAT asks whether attention can become the edge-local message weighting rule.

This changes attention from:

$$
\text{all tokens attend to all tokens}
$$

to:

$$
\text{each node attends only to its graph neighbors}
$$

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

The useful narrowed claim is:

$$
\text{graph neighborhood}
+
\text{learned attention weights}
\rightarrow
\text{adaptive local message passing}
$$

This is not global Transformer attention. The graph still decides which nodes can communicate in one layer.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | graph $G=(V,E)$ with node features |
| Output | updated node representations or node labels |
| Main operation | edge-local attention-weighted neighbor aggregation |
| Attention domain | $\mathcal{N}(i)$, not all nodes unless graph is dense |
| Natural task in paper | node classification |
| Setting | transductive citation graphs and inductive PPI graphs |

For node features:

$$
h_i
\in
\mathbb{R}^{F}
$$

the layer returns:

$$
h_i'
\in
\mathbb{R}^{F'}
$$

using messages from neighboring nodes.

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

## Edge-Local Attention

The attention score is computed from the transformed source and target node features:

$$
e_{ij}
=
\operatorname{LeakyReLU}
\left(
a^\top [Wh_i \,\Vert\, Wh_j]
\right)
$$

where $\Vert$ denotes concatenation.

The key mask is graph structure:

$$
j \in \mathcal{N}(i)
$$

The softmax is local to one node's neighborhood:

$$
\alpha_{ij}
=
\operatorname{softmax}_{j \in \mathcal{N}(i)}(e_{ij})
$$

This differs from sequence attention, where the candidate keys are usually all positions allowed by a mask.

## Multi-Head Graph Attention

GAT uses multiple attention heads.

For intermediate layers, heads are concatenated:

$$
h_i'
=
\mathop{\Vert}_{k=1}^{K}
\sigma
\left(
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij}^{k} W^{k} h_j
\right)
$$

For the final layer, heads can be averaged:

$$
h_i'
=
\sigma
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij}^{k} W^{k}h_j
\right)
$$

Multi-head attention stabilizes learning and lets different heads attend to different neighbor patterns.

## Message Passing View

GAT is a message-passing neural network layer.

| Part | GAT Form |
| --- | --- |
| message | $m_{ij}=W h_j$ |
| edge weight | $\alpha_{ij}$ from attention |
| aggregation | $\sum_{j\in\mathcal{N}(i)} \alpha_{ij}m_{ij}$ |
| update | nonlinearity and optional multi-head merge |

The edge set still controls information flow:

$$
\text{available messages for node } i
=
\{h_j : j \in \mathcal{N}(i)\}
$$

So graph construction remains part of the architecture contract.

## Relation to GCN

| Axis | GCN | GAT |
| --- | --- | --- |
| neighbor weights | fixed by graph degree normalization | learned from node features |
| aggregation | normalized sum | attention-weighted sum |
| graph structure | defines neighbors and weights | defines neighbors; features define weights |
| inductive use | possible but original GCN focus is transductive citation graphs | explicitly tested on PPI inductive setting |

The main move is from structural weighting to feature-dependent weighting.

## Relation to Transformer Attention

Transformer attention:

$$
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d}}
\right)
$$

GAT attention:

$$
\alpha_{ij}
=
\operatorname{softmax}_{j \in \mathcal{N}(i)}
\left(
a^\top[Wh_i\Vert Wh_j]
\right)
$$

The similarity is learned pairwise weighting. The difference is the graph mask and the additive attention form used in the original GAT layer.

## Complexity

For $|V|$ nodes, $|E|$ edges, hidden dimension $F'$, and $K$ heads, GAT computation scales with edges:

$$
O(K |E| F')
$$

not with all possible node pairs:

$$
O(|V|^2)
$$

unless the graph is dense.

This is why GAT is a sparse attention architecture: graph structure limits the attention candidate set.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Attention over graph neighborhoods improves graph learning | citation network and protein-protein interaction benchmarks | benchmark scale and graph construction are limited |
| GAT handles inductive graph settings | PPI experiment uses unseen graphs at test time | not all graph domains are inductive in the same way |
| Learned neighbor weights are useful | comparisons to GCN-style methods | attention weights are not automatically explanations |

## Benchmark Reading

The paper mixes transductive and inductive graph settings.

| Setting | Meaning |
| --- | --- |
| transductive citation graphs | train/test nodes are in the same graph |
| inductive PPI graphs | test graphs are unseen during training |

This distinction matters. A model that works on a single citation graph is not automatically proven to generalize to new graphs, molecules, proteins, or interaction networks.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | node classification |
| Input/output unit | graph with node features to node labels |
| Main datasets | Cora, Citeseer, Pubmed, PPI |
| Main comparison | graph convolution and graph neural baselines |
| Not directly tested | molecular docking, 3D equivariant modeling, dynamic graphs |

## Ablation Reading

| Axis | What it tests | Reading |
| --- | --- | --- |
| attention vs fixed aggregation | learned neighbor weighting | central architecture contribution |
| multi-head attention | stability and capacity | similar motivation to Transformer heads |
| transductive vs inductive tasks | generalization setup | must not be conflated |
| graph sparsity/degree | computational behavior | high-degree nodes increase cost |
| attention visualization | interpretability of neighbor weights | useful but not causal proof |

The strongest reusable idea is not the exact additive attention formula. It is masked, edge-local attention as a graph aggregation rule.

## Graph Construction Risks

GAT can only attend over edges in the graph. If the graph is wrong, missing, noisy, or leakage-prone, attention cannot fully fix that.

| Risk | Consequence |
| --- | --- |
| missing edges | useful neighbors unavailable |
| spurious edges | noisy messages become candidates |
| leakage edges | test information can flow through graph structure |
| high-degree hubs | attention becomes expensive and hard to interpret |
| feature leakage | node features may encode labels indirectly |

For molecular and biological graphs, graph construction is often as important as the GNN layer.

## Implementation Notes

- Attention softmax must be grouped by destination node neighborhood.
- Self-loops should be handled deliberately; they change whether a node preserves its own features directly.
- Sparse batching and edge indexing dominate practical implementation complexity.
- Attention coefficients should not be treated as explanations without separate validation.
- Inductive evaluation requires graph-level splits, not only random node splits.
- Edge features are not central in the original GAT formulation, but many later graph attention models add them.

## Limitations

- Attention is restricted to observed graph neighborhoods, so graph construction still defines the available information.
- Attention coefficients can be unstable or misleading if interpreted as causal explanations.
- Dense high-degree graphs can make neighbor attention expensive.
- Later graph transformers and message-passing variants changed global attention, edge features, positional encodings, and scalability.
- The original benchmarks are small by modern graph-learning standards.
- The layer is not equivariant to 3D geometry unless geometric features or equivariant structure are added.
- Node classification evidence does not directly prove graph-level molecular or protein performance.

## Why It Matters

GAT is a key bridge between [[concepts/architectures/attention|Attention]] and [[concepts/architectures/gnn|Graph neural networks]]. It shows how attention can become a local graph message-passing rule rather than a dense sequence operation.

The reusable pattern is:

$$
\text{graph edges}
\rightarrow
\text{masked attention candidates}
\rightarrow
\text{adaptive message aggregation}
$$

This pattern later appears in graph Transformers, molecular graph networks, knowledge-graph models, and protein interaction modeling.

## Connections

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/modalities/graph|Graph]]
- [[concepts/tasks/graph-prediction|Graph prediction]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[papers/architectures/gcn|Semi-Supervised Classification with GCNs]]
- [[papers/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/index|Architecture papers]]
