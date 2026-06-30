---
title: How Powerful are Graph Neural Networks?
aliases:
  - papers/architectures/gin
  - papers/gin
  - papers/graph-isomorphism-network
  - papers/how-powerful-are-graph-neural-networks
tags:
  - papers
  - architectures
  - graph-neural-network
---

# How Powerful are Graph Neural Networks?

> The paper connects message-passing GNN expressivity to the Weisfeiler-Lehman graph isomorphism test and proposes Graph Isomorphism Network as a maximally expressive 1-WL-style GNN.

## Metadata

| Field | Value |
| --- | --- |
| Paper | How Powerful are Graph Neural Networks? |
| Authors | Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka |
| Year | 2018 preprint; 2019 conference |
| Venue | ICLR 2019 |
| arXiv | [1810.00826](https://arxiv.org/abs/1810.00826) |
| OpenReview | [ryGs6iA5Km](https://openreview.net/forum?id=ryGs6iA5Km) |
| Code | [weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns) |
| Status | full note started |

## One-Line Takeaway

For standard message-passing GNNs, the aggregation and readout functions set an expressive ceiling; sum aggregation plus MLPs can match the discriminative power of the 1-WL test.

## Question

Many GNNs can be written as neighborhood aggregation:

$$
h_v^{(k)}
=
\operatorname{UPDATE}^{(k)}
\left(
h_v^{(k-1)},
\operatorname{AGGREGATE}^{(k)}
\left(
\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}
\right)
\right).
$$

This form is permutation equivariant, but permutation equivariance alone does not say how much graph structure the model can distinguish.

The architecture question is:

$$
\text{When do two different graph neighborhoods collapse to the same embedding?}
$$

The paper answers by comparing GNN aggregation to [[concepts/architectures/wl-test|1-WL color refinement]].

## Main Claim

The expressive power of a message-passing GNN depends on whether its aggregation and readout functions are injective over multisets.

GIN uses:

$$
h_v^{(k)}
=
\operatorname{MLP}^{(k)}
\left(
(1+\epsilon^{(k)})h_v^{(k-1)}
+
\sum_{u\in \mathcal{N}(v)}
h_u^{(k-1)}
\right)
$$

and graph-level pooling:

$$
h_G
=
\operatorname{CONCAT}
\left(
\operatorname{READOUT}
\left(
\{h_v^{(k)} : v\in G\}
\right)
\mid
k=0,\ldots,K
\right).
$$

The central claim:

$$
\text{injective multiset aggregation}
\approx
\text{1-WL expressive power}
$$

within the message-passing family.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | graph $G=(V,E)$ with node features or node labels |
| Hidden unit | node embedding $h_v^{(k)}$ |
| Mixing rule | local neighborhood message passing |
| Aggregator | sum over neighbor embeddings |
| Update | MLP over self state plus summed neighbor state |
| Readout | graph-level pooling over node embeddings from multiple layers |
| Symmetry | permutation equivariant node updates, permutation invariant graph output |
| Reference test | 1-dimensional Weisfeiler-Lehman color refinement |
| Main output setting | graph classification |

The note is important because it turns a vague GNN design question into a testable contract:

$$
\operatorname{AGGREGATE}
:
\{\!\{\mathbb{R}^d\}\!\}
\rightarrow
\mathbb{R}^{d'}
$$

should avoid collapsing different multisets that matter for the task.

## WL Color Refinement

The 1-WL test updates each node color by hashing its previous color together with the multiset of neighbor colors:

$$
c_v^{(k)}
=
\operatorname{HASH}
\left(
c_v^{(k-1)},
\{\!\{c_u^{(k-1)} : u\in \mathcal{N}(v)\}\!\}
\right).
$$

If two nodes have different rooted subtree structures, repeated WL updates can assign them different colors. The key operation is not averaging. It is injective hashing over a multiset.

A message-passing GNN is similar:

$$
h_v^{(k)}
=
\phi^{(k)}
\left(
h_v^{(k-1)},
\{\!\{h_u^{(k-1)} : u\in \mathcal{N}(v)\}\!\}
\right),
$$

but $\phi$ is implemented with differentiable neural functions rather than a discrete hash.

## Why Mean And Max Lose Information

Many aggregators are permutation invariant, but not all are injective over multisets.

Mean aggregation:

$$
\operatorname{mean}(\{1,1,3\})
=
\operatorname{mean}(\{1,3,3\})
$$

can collapse different count patterns.

Max aggregation:

$$
\operatorname{max}(\{1,2,2\})
=
\operatorname{max}(\{2\})
$$

can discard multiplicity and weaker signals.

Sum aggregation can preserve multiplicity through magnitude:

$$
\operatorname{sum}(\{1,1,3\})
\neq
\operatorname{sum}(\{1,3,3\}).
$$

The architectural point is not that sum is always empirically best. It is that sum is the right default when the model needs to distinguish multisets.

## GIN Layer

The GIN update is:

$$
h_v^{(k)}
=
\operatorname{MLP}^{(k)}
\left(
(1+\epsilon^{(k)})h_v^{(k-1)}
+
\sum_{u\in \mathcal{N}(v)}
h_u^{(k-1)}
\right).
$$

Here:

| Symbol | Meaning |
| --- | --- |
| $h_v^{(k)}$ | node state of node $v$ after layer $k$ |
| $\mathcal{N}(v)$ | neighbors of node $v$ |
| $\epsilon^{(k)}$ | scalar controlling self-feature weight |
| $\operatorname{MLP}^{(k)}$ | learnable injective-ish transformation |

If $\epsilon=0$, the self state and neighbor sum are weighted equally:

$$
h_v^{(k)}
=
\operatorname{MLP}^{(k)}
\left(
h_v^{(k-1)}
+
\sum_{u\in \mathcal{N}(v)}
h_u^{(k-1)}
\right).
$$

If $\epsilon$ is learned, the layer can tune the separation between the center node and its neighborhood:

$$
(1+\epsilon^{(k)})h_v^{(k-1)}
$$

acts like a self-loop weight.

## Graph-Level Readout

For graph classification, node embeddings must be pooled into a graph embedding:

$$
h_G^{(k)}
=
\operatorname{READOUT}^{(k)}
\left(
\{h_v^{(k)} : v\in G\}
\right).
$$

GIN concatenates readouts across layers:

$$
h_G
=
\operatorname{CONCAT}
\left(
h_G^{(0)}, h_G^{(1)}, \ldots, h_G^{(K)}
\right).
$$

This matters because different layers represent different neighborhood radii. Layer $k$ roughly captures information from the $k$-hop neighborhood:

$$
h_v^{(k)}
\approx
\text{representation of rooted subtree around }v\text{ up to depth }k.
$$

Concatenating readouts avoids forcing all graph evidence to survive only in the final layer.

## Relation To Earlier GNNs

| Model Family | Aggregation Style | Expressivity Reading |
| --- | --- | --- |
| [[papers/architectures/gcn|GCN]] | normalized neighborhood averaging | stable and simple, but degree/count information can be smoothed |
| GraphSAGE mean | mean aggregation | inductive and scalable, but non-injective over multisets |
| GraphSAGE pooling | learned transform plus max pooling | adaptive, but max can discard multiplicity |
| GIN | sum aggregation plus MLP | designed to match 1-WL discriminative power |

This does not make GIN universally better. It makes the design tradeoff explicit:

$$
\text{stability and normalization}
\quad
\text{vs}
\quad
\text{multiset expressivity}.
$$

## Why It Belongs In Architecture Papers

This paper is not just another graph benchmark paper. It gives a durable rule for reading graph architectures:

| Design Choice | Question To Ask |
| --- | --- |
| aggregation | does it distinguish different multisets? |
| self-loop handling | can the center node be separated from neighbors? |
| update function | is it expressive enough after aggregation? |
| readout | does graph pooling preserve counts and layer-wise evidence? |
| graph construction | are important relations present before message passing starts? |

The paper also explains why GNN papers should report more than accuracy. If the model claims better structure reasoning, the note should ask whether the gain is caused by architecture, features, graph construction, or benchmark split.

## Evidence Pattern

The paper combines theory and experiments:

| Evidence | What It Supports |
| --- | --- |
| WL comparison | message-passing GNNs are bounded by 1-WL-style refinement |
| non-injective aggregator examples | mean/max can collapse distinct graph neighborhoods |
| GIN construction | sum plus MLP can reach maximum 1-WL-level power in the studied family |
| graph classification benchmarks | the expressivity analysis is practically relevant |

The strongest part is the conceptual framework. The benchmark part should be read as evidence that the framework matters, not as proof that GIN dominates every molecular, social, or biological graph task.

## Practical Reading For Molecular Graphs

For molecular modeling, the paper gives a useful warning:

$$
\text{molecule}
\rightarrow
\text{graph}
\rightarrow
\text{message passing}
$$

only works as well as both the graph construction and the aggregation contract.

Questions to ask:

| Question | Why |
| --- | --- |
| Are atoms initialized with enough chemical labels? | WL refinement starts from initial colors/features |
| Are bonds, distances, charges, stereochemistry, or conformers represented? | missing edge information cannot be recovered by aggregation |
| Is the task graph-level, node-level, edge-level, or pair-level? | readout must match target unit |
| Does mean pooling erase degree or count signals? | molecule properties often depend on multiplicity |
| Does the split test scaffold generalization? | graph classification can overstate performance under easy splits |

For structure-based modeling, a scalar 1-WL-style GNN may be insufficient when outputs transform with 3D rotations and translations. Coordinate or force prediction often needs [[concepts/geometric-deep-learning/equivariant-gnn|equivariant GNNs]].

## Limits

GIN is maximally expressive only relative to the studied message-passing family and the 1-WL reference. It does not solve all graph distinguishability.

Important limits:

- 1-WL itself cannot distinguish all non-isomorphic graphs.
- More expressive aggregation can overfit small graph benchmarks.
- Sum aggregation changes scale with graph degree and graph size.
- Initial node and edge features still matter.
- Graph construction can dominate architecture.
- Long-range dependencies may need virtual nodes, positional encodings, higher-order message passing, graph Transformers, or subgraph methods.

The practical conclusion:

$$
\text{GIN is a baseline for expressivity, not a universal default.}
$$

## What To Remember

- Message-passing GNNs recursively aggregate neighbor multisets.
- Expressivity depends on whether aggregation and readout are injective over multisets.
- Sum aggregation preserves multiplicity better than mean or max.
- GIN uses sum aggregation plus MLPs to match 1-WL power in the analyzed setting.
- 1-WL is a useful ceiling, but not a complete graph reasoning standard.
- For molecular and protein graphs, graph construction and split design are as important as the GNN layer.

## Links

- [[concepts/architectures/gnn|Graph Neural Networks]]
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[papers/architectures/gcn|Semi-Supervised Classification with GCNs]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/graphormer|Graphormer]]
