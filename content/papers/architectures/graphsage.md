---
title: Inductive Representation Learning on Large Graphs
aliases:
  - papers/graphsage
  - papers/architectures/graphsage
  - papers/inductive-representation-learning-on-large-graphs
tags:
  - papers
  - architectures
  - graph-neural-network
---

# Inductive Representation Learning on Large Graphs

> The paper introduced GraphSAGE, a sample-and-aggregate framework for inductive node representation learning on large graphs.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Inductive Representation Learning on Large Graphs |
| Authors | William L. Hamilton, Rex Ying, Jure Leskovec |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1706.02216](https://arxiv.org/abs/1706.02216) |
| NeurIPS | [paper page](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs) |
| Project | [SNAP GraphSAGE](https://snap.stanford.edu/graphsage/) |
| Status | full note started |

## One-Line Takeaway

GraphSAGE changes graph representation learning from "learn an embedding table for known nodes" to "learn a neighborhood aggregation function that can embed unseen nodes and graphs."

## Question

Earlier graph embedding methods often learned a vector for each node in one fixed graph:

$$
v_i \mapsto z_i.
$$

That works in a transductive setting, but it does not naturally handle a new node, a growing graph, or an unseen graph without retraining or re-optimizing node embeddings.

GraphSAGE asks:

$$
\text{Can a graph model learn a reusable function } f_\theta(v, G, X)
\text{ that generates node embeddings from features and neighborhoods?}
$$

The architecture answer is sample-and-aggregate message passing.

## Main Claim

An inductive graph encoder can be built by repeatedly sampling fixed-size neighborhoods and aggregating neighbor features with learnable aggregator functions.

At layer $k$, the core pattern is:

$$
h_{\mathcal{N}(v)}^{(k)}
=
\operatorname{AGGREGATE}^{(k)}
\left(
\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}
\right)
$$

$$
h_v^{(k)}
=
\sigma
\left(
W^{(k)}
\cdot
\operatorname{CONCAT}
\left(
h_v^{(k-1)}, h_{\mathcal{N}(v)}^{(k)}
\right)
\right).
$$

After normalization:

$$
h_v^{(k)}
\leftarrow
\frac{h_v^{(k)}}{\lVert h_v^{(k)} \rVert_2}.
$$

The durable contribution is:

$$
\text{node features}
+
\text{sampled neighborhood aggregation}
\rightarrow
\text{inductive graph representation}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | graph $G=(V,E)$, node features $x_v$, target node or batch of target nodes |
| Output | node embedding $z_v=h_v^{(K)}$ and optional node classifier |
| Core operation | sample neighbors, aggregate features, concatenate self state and neighbor state |
| Learnable part | aggregator parameters and layer projections $W^{(k)}$ |
| Non-learnable part | graph adjacency and sampling fanout |
| Main setting | inductive node representation learning |
| Symmetry | permutation-invariant neighbor aggregation, permutation-equivariant node update |
| Scaling lever | fixed fanout sampling controls computation per batch |

The input node can be unseen during training as long as its features and local graph neighborhood are available at inference time.

## Sampling View

Full neighborhood aggregation can become expensive when node degrees are large. GraphSAGE samples a fixed-size neighborhood at each layer:

$$
\mathcal{S}^{(k)}(v)
\subseteq
\mathcal{N}(v).
$$

Then the layer uses:

$$
h_{\mathcal{S}(v)}^{(k)}
=
\operatorname{AGGREGATE}^{(k)}
\left(
\{h_u^{(k-1)} : u \in \mathcal{S}^{(k)}(v)\}
\right).
$$

If each layer samples $S_k$ neighbors, then the number of nodes touched for one root node scales roughly as:

$$
O\left(\prod_{k=1}^{K} S_k\right)
$$

instead of depending on all nodes within the $K$-hop ego network.

This is why GraphSAGE is an architecture paper as much as a graph embedding paper: it makes minibatch training practical for large graphs by defining a fixed computation budget around target nodes.

## Aggregator Families

GraphSAGE studies several neighborhood aggregators. They differ in how they summarize the multiset of neighbor states.

### Mean Aggregator

The mean aggregator averages neighbor embeddings:

$$
h_{\mathcal{N}(v)}^{(k)}
=
\frac{1}{|\mathcal{N}(v)|}
\sum_{u\in\mathcal{N}(v)}
h_u^{(k-1)}.
$$

The node update then concatenates the node's previous state with the aggregated neighbor state:

$$
h_v^{(k)}
=
\sigma
\left(
W^{(k)}
\left[
h_v^{(k-1)}
\Vert
h_{\mathcal{N}(v)}^{(k)}
\right]
\right).
$$

Mean aggregation is simple and stable, but it can collapse different multisets with the same average. This limitation connects to later expressivity analysis in [[papers/architectures/graph-isomorphism-network|GIN]].

### Pooling Aggregator

The pooling aggregator first transforms each neighbor independently:

$$
\tilde{h}_u^{(k)}
=
\sigma
\left(
W_{\text{pool}}h_u^{(k-1)} + b_{\text{pool}}
\right),
$$

then applies an elementwise max:

$$
h_{\mathcal{N}(v)}^{(k)}
=
\max_{u\in\mathcal{N}(v)}
\tilde{h}_u^{(k)}.
$$

This lets the model learn nonlinear feature detectors before pooling. The tradeoff is that max pooling can ignore multiplicity and weaker but repeated signals.

### LSTM Aggregator

The LSTM aggregator processes a random permutation of neighbor states:

$$
h_{\mathcal{N}(v)}^{(k)}
=
\operatorname{LSTM}
\left(
\operatorname{perm}
\left(
\{h_u^{(k-1)} : u\in\mathcal{N}(v)\}
\right)
\right).
$$

This aggregator is expressive, but the LSTM is not inherently permutation invariant. Random neighbor order is used in practice, so the model can work empirically but its symmetry contract is weaker than mean or pooling.

## Inductive vs Transductive

The key distinction is whether the model learns node IDs or a reusable embedding function.

| Axis | Transductive embedding | GraphSAGE |
| --- | --- | --- |
| Learned object | embedding vector per known node | aggregation function shared across nodes |
| New node | requires optimization/retraining or a workaround | can be embedded from features and sampled neighborhood |
| New graph | usually outside the original graph contract | possible if features and graph structure follow the same schema |
| Main dependency | node identity and graph-specific co-occurrence | node features, adjacency, aggregator parameters |
| Scaling path | optimize embeddings for all nodes | minibatch target nodes with sampled fanout |

This does not mean GraphSAGE solves all domain shift. It means the architecture is not tied to a fixed node embedding table.

## Unsupervised Objective

GraphSAGE can be trained with an unsupervised neighborhood-based objective. For a target node $u$, a positive context node $v$ is sampled from nearby nodes, and negative nodes $v_n$ are sampled from a noise distribution.

One common form is:

$$
J_G(z_u)
=
-
\log \sigma(z_u^\top z_v)
-
Q
\cdot
\mathbb{E}_{v_n \sim P_n(v)}
\log \sigma(-z_u^\top z_{v_n}).
$$

where:

| Symbol | Meaning |
| --- | --- |
| $z_u$ | final embedding for node $u$ |
| $v$ | positive node from a neighborhood/context sampling process |
| $v_n$ | negative node |
| $Q$ | number of negative samples |
| $P_n$ | negative sampling distribution |
| $\sigma$ | logistic sigmoid |

The objective encourages nearby nodes to have similar embeddings and random negatives to be separated.

## Supervised Use

For node classification, the final embedding can feed a classifier:

$$
\hat{y}_v
=
\operatorname{softmax}(W_{\text{cls}} z_v + b_{\text{cls}}).
$$

The supervised loss over labeled nodes is:

$$
\mathcal{L}
=
-
\sum_{v\in\mathcal{Y}_L}
\sum_{c=1}^{C}
y_{vc}\log \hat{y}_{vc}.
$$

The important architecture point is that the encoder is still the same sample-and-aggregate function.

## Relation To GCN, GAT, And GIN

| Paper | Main architectural move | What GraphSAGE adds to the shelf |
| --- | --- | --- |
| [[papers/architectures/gcn|GCN]] | normalized full-neighborhood propagation | GraphSAGE makes neighborhood aggregation minibatchable and inductive |
| [[papers/architectures/graph-attention-networks|GAT]] | learned edge-local attention weights | GraphSAGE focuses on sampling and aggregator functions rather than attention weights |
| [[papers/architectures/graph-isomorphism-network|GIN]] | injective multiset aggregation and 1-WL expressivity | GraphSAGE exposes aggregator choices that later expressivity work analyzes |
| [[papers/architectures/graphormer|Graphormer]] | graph structure biases inside Transformer attention | GraphSAGE remains local message passing with sampled neighborhoods |

GraphSAGE is especially useful as the bridge between graph embedding methods and modern minibatch GNN training.

## Evidence

The paper evaluates inductive node classification on evolving information graphs and a protein-protein interaction setting. The important evidence pattern is not just accuracy; it is that the same learned aggregation functions can be applied to nodes or graphs not present during training.

| Evidence Type | What It Supports |
| --- | --- |
| Inductive benchmarks | learned aggregators can generalize beyond the training graph or training nodes |
| Multiple aggregators | architecture quality depends on the aggregation function |
| Sampling design | fixed fanout makes training feasible on large graphs |
| PPI experiment | the architecture can transfer across related graphs rather than only label nodes in one graph |

## Why It Matters

GraphSAGE made three design choices durable:

1. Learn an encoder function, not an embedding table.
2. Use fixed-size neighborhood sampling to make graph minibatches practical.
3. Treat aggregation choice as an architecture component.

Those choices appear throughout later graph learning systems, including recommender GNNs, knowledge graph pipelines, protein interaction models, and large-graph training libraries.

## Limitations

| Limitation | Why It Matters |
| --- | --- |
| Sampling variance | fixed fanout can miss important neighbors, especially for sparse signals |
| Aggregator information loss | mean and max pooling can collapse different neighborhood multisets |
| Shallow neighborhood bias | deeper sampling can grow exponentially or oversmooth representations |
| Feature dependence | inductive generalization assumes useful node features are available for new nodes |
| Graph construction dependence | edges define the accessible context, so bad edges or leakage change the result |
| Domain shift | new graphs must share enough feature and structural semantics with training graphs |

For molecular graphs, the model is a useful general graph baseline, but it does not directly encode continuous 3D geometry. Papers like [[papers/architectures/schnet|SchNet]], [[papers/architectures/dimenet|DimeNet]], and [[papers/architectures/egnn|EGNN]] add geometric inductive bias.

## What To Remember

Use GraphSAGE when the key problem is:

$$
\text{large graph}
+
\text{node features}
+
\text{new nodes or new graphs}
+
\text{local neighborhood signal}.
$$

Do not read it as "the best GNN." Read it as the architecture that made inductive, sampled, minibatch neighborhood aggregation a standard graph learning pattern.

## Links

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]]
- [[papers/architectures/gcn|GCN]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/graph-isomorphism-network|Graph Isomorphism Network]]
- [[papers/architectures/graphormer|Graphormer]]
- [[papers/architectures/neural-message-passing-for-quantum-chemistry|Neural Message Passing for Quantum Chemistry]]
