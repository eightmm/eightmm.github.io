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

Images have grid structure and sequences have order. Graphs have nodes and edges, but no fixed spatial grid or canonical node order. The paper asks how to define a neural layer that mixes node features through graph connectivity while staying simple enough for semi-supervised node classification.

The architecture question is:

$$
\text{How can a neural layer update node states using both node features and graph adjacency while respecting node permutation?}
$$

The paper answers with a first-order graph convolution layer that repeatedly propagates and transforms node features over a normalized adjacency matrix.

## Main Claim

A first-order approximation to spectral graph convolution yields an efficient neural layer for semi-supervised node classification on graph-structured data.

The canonical GCN layer is:

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

where:

| Symbol | Meaning |
| --- | --- |
| $A$ | graph adjacency matrix |
| $I$ | identity matrix for self-loops |
| $\tilde{A}=A+I$ | adjacency with self-loops |
| $\tilde{D}_{ii}=\sum_j\tilde{A}_{ij}$ | degree matrix of $\tilde{A}$ |
| $H^{(\ell)}$ | node features at layer $\ell$ |
| $W^{(\ell)}$ | learned feature projection |
| $\sigma$ | activation function |

The durable claim is not just a benchmark result. It is that graph neural layers can be written as:

$$
\text{feature projection}
+
\text{normalized neighborhood aggregation}
\Rightarrow
\text{node representations that share information over graph edges}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | graph $G=(V,E)$, node feature matrix $X$, adjacency matrix $A$ |
| Output | node-level class probabilities or node embeddings |
| Node unit | document, entity, molecule atom, residue, abstract graph node, depending on dataset |
| Edge unit | graph relation supplied by data or construction protocol |
| Token mixing | neighborhood aggregation through normalized adjacency |
| Learnable part | per-layer feature weights $W^{(\ell)}$ |
| Non-learnable part | graph propagation operator from $\tilde{A}$ |
| Symmetry | permutation equivariant node update |
| Main setting | transductive semi-supervised node classification |

For a two-layer GCN used for node classification:

$$
Z
=
\operatorname{softmax}
\left(
\hat{A}
\operatorname{ReLU}
\left(
\hat{A}XW^{(0)}
\right)
W^{(1)}
\right)
$$

where:

$$
\hat{A}
=
\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}.
$$

The supervised loss is applied only on labeled nodes:

$$
\mathcal{L}
=
-
\sum_{i\in \mathcal{Y}_L}
\sum_{c=1}^{C}
Y_{ic}\log Z_{ic}.
$$

The model can still use the full graph structure during propagation, so unlabeled nodes influence representation through edges even when their labels are not used in the loss.

## From Spectral Graph Convolution To GCN

The paper motivates GCN through spectral graph convolution. For an undirected graph, the normalized graph Laplacian is:

$$
L = I_N - D^{-1/2}AD^{-1/2}.
$$

A spectral graph convolution can be written using eigenvectors $U$ of $L$:

$$
g_\theta * x
=
U g_\theta(\Lambda) U^\top x.
$$

Direct spectral convolution is expensive and graph-specific. Prior work used polynomial approximations such as Chebyshev filters:

$$
g_\theta * x
\approx
\sum_{k=0}^{K}
\theta_k T_k(\tilde{L})x.
$$

GCN uses a first-order approximation, then applies a renormalization trick to get a simple stable propagation operator:

$$
\hat{A}
=
\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}.
$$

This converts graph convolution into sparse neighborhood averaging plus a learned feature transform.

## Renormalization Trick

The self-looped normalized adjacency is:

$$
\tilde{A}=A+I.
$$

The corresponding degree matrix is:

$$
\tilde{D}_{ii}=\sum_j \tilde{A}_{ij}.
$$

The propagation operator is:

$$
\hat{A}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}.
$$

Elementwise, the message from node $j$ to node $i$ is scaled by:

$$
\frac{1}{\sqrt{\tilde{d}_i\tilde{d}_j}}.
$$

So a GCN update can be read as:

$$
h_i^{(\ell+1)}
=
\sigma
\left(
\sum_{j\in \mathcal{N}(i)\cup\{i\}}
\frac{1}{\sqrt{\tilde{d}_i\tilde{d}_j}}
h_j^{(\ell)}W^{(\ell)}
\right).
$$

This is the bridge between spectral notation and message passing notation.

## Message Passing View

In [[concepts/architectures/gnn|GNN]] language, GCN is a message-passing layer with fixed degree-normalized messages:

$$
m_{ij}^{(\ell)}
=
\frac{1}{\sqrt{\tilde{d}_i\tilde{d}_j}}
h_j^{(\ell)}W^{(\ell)}
$$

$$
h_i^{(\ell+1)}
=
\sigma
\left(
\sum_{j\in \mathcal{N}(i)\cup\{i\}}
m_{ij}^{(\ell)}
\right).
$$

The aggregation is permutation invariant over neighbors because it is a sum. The node-level function is permutation equivariant:

$$
f(PX, PAP^\top)=P f(X,A)
$$

for a node permutation matrix $P$, assuming features and adjacency are permuted consistently.

## Block View

| Component | Role | Architecture Implication |
| --- | --- | --- |
| Self-loop addition | includes each node's own features | avoids replacing a node only with neighbors |
| Degree normalization | controls scale across high/low degree nodes | prevents raw degree from dominating updates |
| Sparse propagation | mixes local neighborhoods | receptive field grows with depth |
| Linear projection | learns feature transformation | shared across all nodes |
| Nonlinearity | adds expressive depth | usually ReLU in the canonical model |
| Labeled-node loss | trains from partial labels | graph structure propagates information |

The layer is intentionally simple. This simplicity makes GCN a canonical baseline for graph architecture papers.

## Transductive Semi-Supervised Setting

The paper's main setting is transductive node classification. The model sees:

$$
G=(V,E),\quad X,\quad Y_L
$$

where $Y_L$ labels only a subset of nodes. It predicts labels for other nodes in the same graph.

This is different from inductive graph learning:

| Setting | Graph Available During Training | Test Unit |
| --- | --- | --- |
| transductive node classification | full graph including unlabeled test nodes | hidden labels in same graph |
| inductive node/graph classification | training graphs only | new nodes or new graphs |

This distinction matters. A GCN result on citation networks does not automatically imply performance on new unseen graphs, molecular scaffold splits, protein-family splits, or dynamic graphs.

## Evidence Reading

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| GCN works for semi-supervised node classification | citation-network benchmarks | simple normalized propagation is effective in homophilous graphs | benchmark scale is modest |
| First-order graph convolution is efficient | sparse matrix operations | graph propagation can be implemented cheaply | assumes sparse graph and fixed adjacency |
| Graph structure improves classification | comparisons with feature-only and graph baselines | edges carry useful label signal | only when graph edges align with task |
| Two-layer GCN is strong enough for benchmarks | reported accuracy with shallow model | local neighborhoods can propagate labels | deeper GCNs can oversmooth |

The evidence should be read as a graph-learning baseline paper, not as a universal graph reasoning solution.

## GCN vs GAT

[[papers/architectures/graph-attention-networks|Graph Attention Networks]] replace fixed degree-normalized neighbor weights with learned attention weights.

| Dimension | GCN | GAT |
| --- | --- | --- |
| Neighbor weights | fixed by graph degree | learned from node features |
| Aggregation | normalized sum | attention-weighted sum |
| Edge adaptivity | low | higher |
| Parameters in mixing | feature projection only | projection plus attention scoring |
| Interpretability | degree-normalized smoothing | attention distribution can be inspected, with caveats |
| Baseline role | simple graph convolution | adaptive graph message passing |

GCN asks: "what if every neighbor contributes through a normalized graph operator?" GAT asks: "what if the model learns which neighbors matter?"

## GCN vs Graph Transformer

A graph Transformer often allows broader or fully connected attention with graph/edge biases. GCN keeps the graph sparse and local:

$$
\operatorname{GCN}: \quad j\in\mathcal{N}(i)
$$

whereas attention-heavy graph models may use:

$$
j\in V
$$

or combine local edges with global tokens. The tradeoff is:

- GCN is cheap and strongly tied to given edges;
- graph Transformers can model long-range dependencies but need positional/edge biases and more compute;
- dense attention can blur the distinction between graph structure and learned relation structure.

## Receptive Field And Depth

One GCN layer mixes one-hop neighborhoods. Two layers mix two-hop neighborhoods:

$$
H^{(2)}
\sim
\hat{A}^2 X W
$$

Ignoring nonlinearities and weights, deeper stacks repeatedly apply $\hat{A}$:

$$
H^{(L)} \sim \hat{A}^{L}X.
$$

This explains both the benefit and the risk. More layers enlarge the receptive field, but repeated smoothing can make node representations too similar.

## Oversmoothing

Repeated normalized propagation can make node features converge toward a low-frequency subspace over the graph:

$$
\hat{A}^{L}X
\quad\text{becomes increasingly smooth as}\quad L\to\infty.
$$

In practice, oversmoothing appears when node embeddings from different classes become hard to distinguish after too many message-passing layers.

Checks:

- compare node embedding variance across layers;
- test shallow vs deep models under same training budget;
- add residual connections, normalization, jumping knowledge, or attention only if the task needs depth;
- inspect whether the graph has long-range dependencies or just local homophily.

## Oversquashing

Oversquashing occurs when information from many distant nodes must pass through small graph bottlenecks. GCN's local aggregation can struggle when relevant signals are far away and the graph has narrow cuts.

This is separate from oversmoothing:

| Failure | Mechanism | Symptom |
| --- | --- | --- |
| oversmoothing | repeated averaging makes embeddings similar | class separation collapses |
| oversquashing | too much information compressed through limited edges | long-range dependencies fail |

GCN is a useful baseline partly because these failures are easy to reason about.

## Homophily Assumption

GCN performs well when neighboring nodes tend to share labels or useful information:

$$
(i,j)\in E
\Rightarrow
y_i \approx y_j
$$

This is common in citation networks where related documents link to each other. But in heterophilous graphs, neighbors may systematically differ:

$$
(i,j)\in E
\Rightarrow
y_i \ne y_j
$$

In such cases, naive smoothing can hurt. A paper claiming improved graph architecture should state whether the benchmark is homophilous, heterophilous, molecular, geometric, temporal, or knowledge-graph-like.

## Graph Construction Boundary

GCN assumes the adjacency matrix is meaningful. But [[concepts/architectures/graph-construction|graph construction]] is often a modeling choice.

For a constructed graph:

$$
A = C_\psi(u)
$$

where $u$ is the raw object and $C_\psi$ is the construction rule. Examples:

- citation links in a document graph;
- chemical bonds in a molecular graph;
- residue contacts in a protein graph;
- radius or kNN edges from coordinates;
- similarity edges from embeddings.

The GCN layer only defines propagation after the graph exists. If graph construction changes, the architecture comparison changes too.

## Molecular And Structural Modeling Reading

GCN is often a starting point for molecular graph models, but the canonical paper is not a molecular modeling paper.

For molecular use, ask:

- Are nodes atoms, residues, conformers, pockets, or abstract entities?
- Are edges covalent bonds, distance cutoffs, kNN edges, contacts, or interactions?
- Are edge attributes used, or only adjacency?
- Is the task graph-level, node-level, edge-level, or complex-level?
- Does the model need 3D equivariance?
- Are scaffold, protein-family, or complex splits controlled?

Plain GCN on a bond graph is permutation equivariant, but not rotation equivariant. If the target depends on 3D coordinates, a scalar GCN may be insufficient without geometric features or equivariant architecture.

## Implementation Notes

The canonical layer can be implemented as sparse matrix multiplication:

$$
H' = \hat{A}HW.
$$

Practical details:

| Detail | Why It Matters |
| --- | --- |
| self-loops | controls whether node identity is retained |
| degree normalization | changes scale and high-degree behavior |
| sparse format | affects memory and throughput |
| feature dropout | regularizes node features |
| edge dropout | regularizes graph propagation |
| cached adjacency | valid only if graph is fixed |
| normalization | BatchNorm/LayerNorm/GraphNorm choices change training |
| split unit | transductive vs inductive evaluation |

If $\hat{A}$ is precomputed, ensure it is computed from the allowed graph only. In transductive settings, using the full graph is part of the benchmark; in inductive settings, using test graph structure during training can be leakage.

## Common Misreadings

### "GCN is just CNN on graphs"

GCN is inspired by graph spectral convolution, but it is not a direct translation of image convolution. Image CNNs use local grid kernels with ordered positions. GCN uses unordered neighbor aggregation through graph edges.

### "GCN proves graph structure is always useful"

Only if the graph edges carry relevant signal. Bad graph construction can make propagation harmful.

### "More GCN layers always mean more reasoning"

More layers increase receptive field but can cause oversmoothing and oversquashing.

### "Citation benchmark success transfers to molecular modeling"

Not automatically. Molecular graphs have chemistry, geometry, conformers, scaffold splits, and often graph-level targets.

## What To Check In Later Graph Papers

- Is the baseline a properly tuned GCN?
- Is the graph construction identical across baselines?
- Are node/edge features identical?
- Is the task transductive or inductive?
- Are splits created before graph construction?
- Does the method address oversmoothing, oversquashing, heterophily, or long-range dependencies?
- Does the reported gain come from architecture, features, data split, or training budget?
- Are self-loops and normalization treated consistently?
- Is the benchmark homophilous?

## Why It Still Matters

GCN remains the cleanest reference point for graph architecture notes:

- it defines a concrete node-message-passing layer;
- it connects spectral graph convolution to message passing;
- it shows why degree normalization matters;
- it makes semi-supervised node classification a standard graph-learning task;
- it provides a baseline for [[papers/architectures/graph-attention-networks|GAT]], graph Transformers, molecular GNNs, and geometric models.

For this wiki, GCN should sit next to [[concepts/architectures/gnn|Graph neural networks]], [[concepts/architectures/graph-construction|Graph construction]], [[papers/architectures/graph-attention-networks|Graph Attention Networks]], and [[concepts/geometric-deep-learning/index|Geometric deep learning]].

## Limitations

- The canonical benchmarks are small compared with modern graph workloads.
- The main setting is transductive node classification.
- Fixed degree-normalized aggregation cannot adaptively choose neighbors.
- Performance depends strongly on graph quality and homophily.
- Stacking many layers can cause oversmoothing.
- Local message passing can suffer from oversquashing.
- Plain GCN does not directly model 3D equivariance.
- Molecular and protein applications need additional checks for graph construction, splits, and coordinate availability.

## Connections

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/index|Architecture papers]]
