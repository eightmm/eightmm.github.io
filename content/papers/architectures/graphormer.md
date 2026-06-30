---
title: Graphormer
aliases:
  - papers/graphormer
  - papers/do-transformers-really-perform-bad-for-graph-representation
tags:
  - papers
  - architectures
  - graph-transformer
  - graph-neural-networks
---

# Graphormer

> The paper shows that a standard Transformer can become a strong graph representation model when graph structure is encoded into attention and node inputs.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Do Transformers Really Perform Bad for Graph Representation? |
| Authors | Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu |
| Year | 2021 |
| Venue | NeurIPS 2021 |
| arXiv | [2106.05234](https://arxiv.org/abs/2106.05234) |
| Proceedings | [NeurIPS 2021 paper](https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf) |
| Project | [Microsoft Research Graphormer](https://www.microsoft.com/en-us/research/project/graphormer/) |
| Status | full note started |

## One-Line Takeaway

Graphormer keeps the global self-attention style of Transformers but injects graph structure through centrality, shortest-path distance, and edge encodings.

## Question

Transformers are strong for sequences because token identity and position are both encoded:

$$
z_i^{(0)} = x_i + p_i.
$$

For a graph:

$$
G=(V,E),
$$

there is no canonical sequence order. A naive Transformer over node tokens:

$$
H' = \operatorname{Transformer}(H)
$$

can attend globally, but it does not automatically know:

- which nodes are adjacent;
- how far apart two nodes are;
- which edge types connect them;
- which nodes are structurally central;
- how graph topology should bias attention.

The paper asks:

> Do Transformers perform poorly on graphs because the architecture is unsuitable, or because graph structure has not been encoded properly?

## Main Claim

Graph Transformers can work well if graph structural information is encoded into the model. Graphormer uses three main graph encodings:

1. centrality encoding for node degree information;
2. spatial encoding for graph distance between nodes;
3. edge encoding for shortest-path edge features.

The attention logit between node $i$ and node $j$ becomes:

$$
A_{ij}
=
\frac{(h_iW^Q)(h_jW^K)^\top}{\sqrt{d}}
+
b_{ij},
$$

where $b_{ij}$ is a graph-structure bias.

Then:

$$
\alpha_{ij}
=
\operatorname{softmax}_j(A_{ij}).
$$

The key idea is simple:

$$
\text{global attention}
+
\text{graph structural bias}
\Rightarrow
\text{graph-aware Transformer}.
$$

## Architecture Contract

| Component | Role |
| --- | --- |
| node embedding | raw atom/node feature representation |
| centrality encoding | degree/importance signal added to node input |
| spatial encoding | shortest-path distance bias in attention |
| edge encoding | edge-feature bias along shortest path |
| graph token | graph-level readout token |
| Transformer blocks | global node-node interaction |

Graphormer is not a local message-passing GNN. It is a dense attention model whose attention scores are biased by graph structure.

## Input Representation

For each node $v_i$, start with a node feature embedding:

$$
x_i \in \mathbb{R}^{d}.
$$

Graphormer adds degree-based centrality encodings. For directed graphs, it can use in-degree and out-degree:

$$
h_i^{(0)}
=
x_i
+
z^{-}_{\deg^-(v_i)}
+
z^{+}_{\deg^+(v_i)}.
$$

For undirected graphs, this reduces to a degree-like centrality signal.

The intuition:

$$
\text{node identity alone}
\neq
\text{node role in graph}.
$$

Degree is a cheap structural prior that helps the Transformer distinguish node roles.

## Spatial Encoding

For nodes $i$ and $j$, define graph distance:

$$
\phi(v_i,v_j)
=
\text{length of shortest path from } v_i \text{ to } v_j.
$$

Graphormer maps this distance to a learned scalar bias:

$$
b_{ij}^{\text{spatial}}
=
\operatorname{Embed}_{\text{dist}}(\phi(v_i,v_j)).
$$

This is added to the attention logit:

$$
A_{ij}
=
\frac{q_i k_j^\top}{\sqrt{d}}
+
b_{ij}^{\text{spatial}}.
$$

So even though every node can attend to every other node, the attention score is aware of graph distance.

This is analogous to relative position bias in sequence Transformers:

$$
\text{sequence relative position}
\rightarrow
\text{graph shortest-path distance}.
$$

## Edge Encoding

If a graph has edge features, such as bond type:

$$
e_{uv},
$$

the shortest path between node $i$ and node $j$ contains edge features:

$$
P_{ij}=(e_1,e_2,\dots,e_{\phi(i,j)}).
$$

Graphormer encodes edge information along this path and adds it to attention:

$$
b_{ij}^{\text{edge}}
=
\operatorname{EdgeEncode}(P_{ij}).
$$

The total bias can be read as:

$$
b_{ij}
=
b_{ij}^{\text{spatial}}
+
b_{ij}^{\text{edge}}.
$$

Then:

$$
A_{ij}
=
\frac{q_i k_j^\top}{\sqrt{d}}
+
b_{ij}.
$$

This makes attention graph-aware without forcing attention to be local.

## Graph Token

For graph-level prediction, Graphormer uses a graph representation token, analogous to a class token:

$$
h_{\text{graph}}.
$$

All nodes can attend to it, and it can attend to all nodes. The final graph-level prediction is:

$$
y = \rho(h_{\text{graph}}^L).
$$

This is a Transformer-style readout rather than a pure sum/mean pooling readout.

## Attention Block

For a single attention head:

$$
Q=HW^Q,
\qquad
K=HW^K,
\qquad
V=HW^V.
$$

Graphormer modifies the logit matrix:

$$
S
=
\frac{QK^\top}{\sqrt{d_h}}
+
B,
$$

where:

$$
B_{ij}=b_{ij}.
$$

Then:

$$
\operatorname{Attn}(H)
=
\operatorname{softmax}(S)V.
$$

So the architecture keeps the standard Transformer core, but the bias matrix $B$ is graph-derived.

## Why It Can Cover Message Passing

Traditional GNN message passing often has the form:

$$
h_i^{l+1}
=
\operatorname{Update}
\left(
h_i^l,
\sum_{j\in\mathcal{N}(i)}
\operatorname{Message}(h_i^l,h_j^l,e_{ij})
\right).
$$

Graphormer has global attention:

$$
h_i^{l+1}
=
\sum_{j\in V}
\alpha_{ij}Vh_j.
$$

If attention biases make non-neighbors negligible and neighbors dominant, the model can emulate local aggregation:

$$
\alpha_{ij}\approx 0
\quad \text{for } j\notin \mathcal{N}(i).
$$

This supports the paper's claim that graph structural encodings allow Graphormer to cover many GNN-like behaviors while retaining global attention.

## Comparison to GCN

| Property | [[papers/architectures/gcn|GCN]] | Graphormer |
| --- | --- | --- |
| Interaction pattern | local adjacency aggregation | global attention with structural bias |
| Graph structure | adjacency normalization | distance/edge/centrality encodings |
| Receptive field | grows with layers | global from one layer |
| Edge features | not central in original GCN | explicitly encoded |
| Readout | pooling after node embeddings | graph token |
| Main risk | oversmoothing/locality limits | quadratic attention and bias design |

Graphormer is not simply a GCN with attention. It changes graph learning from local propagation to globally biased attention.

## Comparison to GAT

| Property | [[papers/architectures/graph-attention-networks|GAT]] | Graphormer |
| --- | --- | --- |
| Attention scope | local neighbors | all node pairs |
| Attention score | learned from node features | node features plus graph structural bias |
| Edge/path encoding | limited in base form | central design element |
| Graph-level tasks | possible with pooling | graph token built in |
| Complexity | tied to edge count | tied to $n^2$ attention |

GAT answers "which neighbor matters?" Graphormer asks "how should every node pair interact given graph structure?"

## Comparison to Sequence Transformer

| Property | Sequence Transformer | Graphormer |
| --- | --- | --- |
| Position | token index | graph distance, centrality, edge path |
| Attention scope | all tokens | all nodes |
| Structure bias | relative/absolute position | shortest-path and edge encodings |
| Readout | class token or sequence output | graph token or node outputs |
| Inductive bias | order | graph topology |

Graphormer is best read as a Transformer with graph-specific positional encoding.

## Complexity

For $n$ nodes, dense attention costs:

$$
O(n^2d).
$$

The structural bias matrix also has pairwise size:

$$
B\in\mathbb{R}^{n\times n}.
$$

This is practical for many molecular graphs, but can be difficult for very large graphs.

Important scaling terms:

- number of nodes $n$;
- number of edges $m$;
- shortest-path preprocessing cost;
- pairwise bias storage;
- attention memory;
- graph batch size.

Graphormer is therefore most natural for graph-level prediction on moderate-size graphs, not huge web-scale node classification without approximation.

## Molecular Graph Reading

Graphormer is directly relevant to molecular property prediction because molecular graphs have:

- atom features;
- bond types;
- graph distances;
- graph-level labels;
- moderate graph sizes in many benchmarks.

For a molecule:

$$
v_i = \text{atom},
\qquad
e_{ij}=\text{bond}.
$$

Centrality can encode atom degree:

$$
\deg(v_i).
$$

Shortest-path distance can encode graph separation:

$$
\phi(v_i,v_j).
$$

Edge encoding can incorporate bond sequences along paths.

This makes Graphormer a strong graph-level molecular baseline when only 2D graph topology is used.

## Where Graphormer Is Not Enough

For structure-based modeling, graph topology is not the whole story. A molecule or protein-ligand complex also has 3D geometry:

$$
x_i \in \mathbb{R}^3.
$$

Graphormer does not automatically guarantee rotation/translation equivariance. If 3D coordinates are important, compare to:

- [[papers/architectures/egnn|E(n) Equivariant GNN]];
- [[papers/architectures/se3-transformer|SE(3)-Transformer]];
- geometric graph models.

Graphormer is mainly a graph topology Transformer. It is not by itself a 3D equivariant architecture.

## Evidence Reading

The paper emphasizes graph-level representation learning and OGB-style benchmarks. The lasting contribution is the structural encoding recipe rather than one specific leaderboard result.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Transformers can work on graphs | graph benchmark performance | graph structure encoding is enough to make Transformer competitive | benchmark and preprocessing dependent |
| Spatial/edge/centrality encodings matter | ablations | structural bias is central to performance | encoding choices are task-specific |
| Graphormer can cover GNN variants | theoretical characterization | attention with proper bias can emulate local message passing | practical training still differs |
| Strong molecular graph performance | OGB molecular tasks | topology-aware Transformer is useful for molecules | 2D graph tasks are not full 3D structure modeling |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | graph-level representation learning |
| Input unit | graph nodes, edges, structural distances |
| Output unit | graph-level prediction |
| Core mechanism | Transformer attention plus graph structural encodings |
| Main comparison | GNNs and graph attention variants |
| Key encodings | centrality, shortest-path distance, edge path |
| Main scaling issue | $O(n^2)$ attention and pairwise bias |
| Not the claim | SE(3)-equivariant 3D molecular modeling |

## Implementation Notes

### Shortest-Path Bias

Precompute shortest-path distance:

$$
D_{ij}=\phi(v_i,v_j).
$$

Then map:

$$
b_{ij}^{\text{spatial}}=\operatorname{Embed}(D_{ij}).
$$

Disconnected node pairs need a special distance bucket.

### Edge Path Encoding

For an edge-feature path:

$$
P_{ij}=(e_1,\dots,e_k),
$$

the encoding must handle path length and edge feature type. If there are multiple shortest paths, implementation choices matter.

### Graph Token Bias

The graph token may need separate bias handling because it is not a normal graph node.

### Batch Padding

Batched graphs need masks so nodes from different graphs do not attend to each other:

$$
A_{ij}=-\infty
\quad \text{if } i,j \text{ belong to different graphs}.
$$

### Large Graphs

For large $n$, dense pairwise attention can dominate:

$$
O(n^2).
$$

Approximation, sampling, locality, or hybrid GNN-Transformer designs may be needed.

## Common Misreadings

### "Graphormer proves raw Transformers understand graphs."

No. It shows Transformers can work when graph structure is explicitly encoded.

### "Shortest-path distance is the same as 3D distance."

No. Shortest-path distance is graph-topological. 3D distance is geometric. Molecular tasks may need both.

### "Graphormer replaces equivariant models."

No. Graphormer is not an equivariant 3D architecture. It is a graph-structure-aware Transformer.

### "Global attention is always better than message passing."

No. Global attention is flexible but expensive and can over-globalize. Local message passing can be better when locality is the right inductive bias.

## Later-Paper Checklist

When reading later graph Transformer papers, ask:

- What graph positional encoding is used?
- Does attention run over all node pairs or only edges?
- Are edge features used directly or through shortest paths?
- Is there a graph token or pooling readout?
- How does complexity scale with node count?
- Are comparisons made against GCN, GAT, and strong message-passing baselines?
- Does the task require 3D geometry or only graph topology?
- Are molecular splits scaffold-aware or otherwise leakage-resistant?
- Are structural encodings ablated separately?

## Why It Matters

Graphormer is an anchor paper because it clarified a practical route for graph Transformers:

$$
\text{Transformer}
+
\text{centrality}
+
\text{spatial distance}
+
\text{edge path bias}.
$$

It belongs next to GCN and GAT because it marks the move from local graph message passing to graph-structured global attention.

For this wiki, it is also a useful bridge between general AI architectures and molecular graph modeling.

## Limitations

- Dense attention is expensive for large graphs.
- Structural encodings require preprocessing and careful batching.
- It is not 3D equivariant.
- Shortest-path topology may miss important geometric or electronic structure.
- Benchmark gains can depend on dataset split, featurization, and graph size.
- Global attention can overfit or over-globalize when locality matters.

## Connections

- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[papers/architectures/gcn|GCN]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]
- [[papers/architectures/index|Architecture papers]]
