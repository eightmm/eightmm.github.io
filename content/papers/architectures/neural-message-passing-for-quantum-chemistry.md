---
title: Neural Message Passing for Quantum Chemistry
aliases:
  - papers/neural-message-passing
  - papers/mpnn
  - papers/neural-message-passing-for-quantum-chemistry
tags:
  - papers
  - architectures
  - gnn
  - molecular-modeling
---

# Neural Message Passing for Quantum Chemistry

> The paper gave molecular graph learning a reusable message-passing architecture language.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Message Passing for Quantum Chemistry |
| Authors | Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl |
| Year | 2017 |
| Venue | ICML 2017 |
| arXiv | [1704.01212](https://arxiv.org/abs/1704.01212) |
| PMLR | [v70/gilmer17a](https://proceedings.mlr.press/v70/gilmer17a.html) |
| Status | verified |

## Question

Molecules are naturally graphs:

$$
G=(V,E)
$$

where atoms are nodes and bonds or learned relations are edges. Before this paper, graph neural models existed under several names and formulations. The paper asks whether many of them can be understood through a common message-passing interface for molecular property prediction.

The architecture question is:

$$
\text{How should a neural network compute over molecular graphs while respecting node-order invariance?}
$$

The paper's answer is the Message Passing Neural Network framework.

## Main Claim

MPNNs separate graph learning into:

1. a message passing phase;
2. a readout phase.

The generic form is:

$$
m_v^{t+1}
=
\sum_{w\in N(v)}
M_t(h_v^t,h_w^t,e_{vw}),
$$

$$
h_v^{t+1}
=
U_t(h_v^t,m_v^{t+1}),
$$

followed by graph-level readout:

$$
\hat y
=
R(\{h_v^T: v\in G\}).
$$

The durable contribution is the interface:

$$
\text{node state}
+
\text{edge-aware messages}
+
\text{permutation-invariant readout}.
$$

This interface became the standard way to describe many GNNs, especially for molecular graphs.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | molecular graph with atom and bond features |
| Node unit | atom |
| Edge unit | bond or graph relation |
| Hidden state | atom representation $h_v^t$ |
| Message | edge-conditioned information from neighbors |
| Aggregation | permutation-invariant sum over neighbors |
| Update | recurrent or feed-forward state update |
| Output | graph-level molecular property |
| Readout | permutation-invariant graph pooling/readout |
| Main domain | quantum chemistry and molecular property prediction |

The contract matters because molecular node order is arbitrary. If atoms are permuted, graph-level predictions should not change:

$$
f(PX, PAP^\top)=f(X,A).
$$

For node-level states, the representation should be permutation equivariant:

$$
H(PX,PAP^\top)=PH(X,A).
$$

## Message Passing Phase

At step $t$, each node $v$ has a hidden state:

$$
h_v^t\in\mathbb{R}^{d}.
$$

Each neighbor $w$ sends a message:

$$
M_t(h_v^t,h_w^t,e_{vw}).
$$

The node aggregates messages:

$$
m_v^{t+1}
=
\sum_{w\in N(v)}
M_t(h_v^t,h_w^t,e_{vw}).
$$

Then it updates its hidden state:

$$
h_v^{t+1}
=
U_t(h_v^t,m_v^{t+1}).
$$

The sum is crucial. It makes the aggregation independent of neighbor order:

$$
\sum_{w\in N(v)} a_w
=
\sum_{w\in \pi(N(v))} a_w.
$$

This is the basic permutation symmetry that makes GNNs valid for unordered graph representations.

## Readout Phase

After $T$ message-passing steps, the model produces a graph-level prediction:

$$
\hat y
=
R(\{h_v^T\}_{v\in V}).
$$

The readout $R$ must be permutation invariant. Common forms include:

$$
R(\{h_v\})=\rho\left(\sum_v \phi(h_v)\right),
$$

or an order-invariant set function.

For molecular property prediction, this means the predicted scalar should not depend on atom indexing:

$$
\hat y(G)
=
\hat y(\pi(G)).
$$

## Why This Is an Architecture Paper

The paper does not introduce only one block. It defines a family:

$$
(M_t,U_t,R)
\quad
\text{define an MPNN}.
$$

Different choices recover or resemble many graph models:

| Component | Design Question |
| --- | --- |
| $M_t$ | what information moves along edges? |
| $U_t$ | how does a node update its state? |
| $R$ | how does the whole graph become a prediction? |
| $T$ | how many graph hops can influence a node? |
| $e_{vw}$ | what edge or relation information is available? |

This is why the paper remains useful even when later GNNs have different names.

## Molecular Graph Semantics

For molecules, a graph is not an abstract network only. Each modeling choice has chemical meaning.

| Graph Element | Molecular Meaning |
| --- | --- |
| node $v$ | atom |
| node feature $h_v^0$ | element, charge, valence, aromaticity, hybridization, etc. |
| edge $(v,w)$ | bond or relation |
| edge feature $e_{vw}$ | bond type, distance, conjugation, ring membership, etc. |
| graph target $y$ | molecular property |

The paper's benchmark context is quantum chemistry, where targets are molecular properties. That makes MPNN a bridge between general graph architecture and molecular modeling.

## Edge-Conditioned Messages

A central design freedom is whether edge features influence messages:

$$
M_t(h_v,h_w,e_{vw}).
$$

In molecular graphs, this matters because a carbon-carbon single bond and an aromatic bond should not necessarily transmit the same information.

An edge-conditioned form can be viewed as:

$$
m_{vw}
=
A(e_{vw})h_w,
$$

where $A(e_{vw})$ is a learned edge-dependent transformation.

This gives the graph architecture a way to use chemical relation types without hard-coding all chemistry.

## Number of Message Passing Steps

After one step, node $v$ sees its immediate neighbors:

$$
N_1(v).
$$

After $T$ steps, it can receive information from a $T$-hop neighborhood:

$$
N_T(v).
$$

This creates a receptive-field interpretation:

$$
\text{GNN depth}
\approx
\text{graph-hop receptive field}.
$$

For molecules, the number of steps controls how far information can travel through the molecular graph.

| Too Few Steps | Too Many Steps |
| --- | --- |
| misses nonlocal molecular context | oversmoothing or noisy long-range mixing |
| limited functional group interaction | more computation and possible over-squashing |

## Relation to GCN

[[papers/architectures/gcn|GCN]] can be read as a specific message-passing model with normalized adjacency aggregation:

$$
H^{(\ell+1)}
=
\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(\ell)}W^{(\ell)}).
$$

MPNN is more general. It emphasizes:

$$
M_t(h_v,h_w,e_{vw})
$$

and therefore fits molecular edge features naturally.

| Axis | GCN | MPNN |
| --- | --- | --- |
| aggregation | normalized adjacency matrix | generic neighborhood message sum |
| edge features | not central in the basic form | central design input |
| domain framing | semi-supervised graph learning | molecular graph property prediction |
| interface | matrix propagation | message, update, readout functions |

## Relation to GAT

[[papers/architectures/graph-attention-networks|GAT]] changes aggregation by learning attention weights:

$$
h_i'
=
\sigma
\left(
\sum_{j\in N(i)}
\alpha_{ij}Wh_j
\right).
$$

This can also be seen as message passing:

$$
M(h_i,h_j,e_{ij})=\alpha_{ij}Wh_j.
$$

The MPNN framework helps place attention-based GNNs in the same family:

$$
\text{attention}
\Rightarrow
\text{learned message weight}.
$$

## Relation to GIN and Expressivity

[[papers/architectures/graph-isomorphism-network|GIN]] later studied the expressive power of message-passing GNNs and related them to the Weisfeiler-Lehman test.

MPNN gives the modeling interface. GIN asks how powerful this interface can be under different aggregation choices.

| Paper | Main Role |
| --- | --- |
| MPNN | defines a general graph message-passing framework for molecular prediction |
| GIN | analyzes expressivity of message-passing aggregation |
| Graphormer | adds Transformer-style global graph biases |

Read MPNN before GIN if the goal is to understand the common GNN API. Read GIN after MPNN if the goal is expressivity.

## Relation to Molecular Modeling

MPNN is especially important for this blog because it connects architecture to molecular objects:

$$
\text{molecule}
\rightarrow
\text{graph}
\rightarrow
\text{message passing}
\rightarrow
\text{property prediction}.
$$

This route is foundational for:

- molecular property prediction;
- molecular representation learning;
- quantum chemistry targets;
- ligand graph encoders;
- protein-ligand interaction graphs;
- geometry-aware molecular GNNs.

It also introduces the key warning:

$$
\text{graph construction is part of the model}.
$$

Changing atom features, edge features, conformer information, or distance edges can change the task.

## Evidence to Read

The paper evaluates neural message passing on quantum chemistry prediction tasks.

Read evidence along these axes:

| Evidence | What It Supports |
| --- | --- |
| QM9-style molecular property prediction | message passing works for molecular graphs |
| comparison to prior graph models | unified framework is competitive |
| edge/message variants | message function choice matters |
| readout variants | graph-level pooling affects property prediction |
| target-specific performance | different chemical properties may require different context |

The important question is not just whether one MPNN variant wins. It is whether the message/update/readout decomposition is a useful architecture abstraction.

## Evaluation Risks

Molecular graph papers are easy to overread. Check:

| Risk | Check |
| --- | --- |
| split leakage | scaffold, molecule identity, conformer, or near-duplicate leakage |
| feature leakage | target-derived or future-derived graph features |
| graph construction confound | bonds only vs distance edges vs conformer geometry |
| readout mismatch | graph-level property hidden in rare substructure |
| target heterogeneity | different quantum properties have different physical locality |
| baseline weakness | compare against strong fingerprints and tuned GNNs |

For later molecular GNN papers, ask whether the claimed architecture improvement is actually a graph construction, feature engineering, data split, or target normalization change.

## Failure Modes and Caveats

- Standard message passing is local by graph hops and can struggle with long-range dependencies.
- Repeated aggregation can cause oversmoothing.
- Information may be oversquashed through narrow graph bottlenecks.
- A 2D molecular graph may miss conformational and stereochemical information unless encoded.
- Message passing is permutation-aware but not automatically rotation/translation equivariant for 3D coordinates.
- Edge construction choices can make comparisons unfair.

## Why This Matters for Architecture Reading

MPNN gives a reusable grammar:

$$
\text{message}
\rightarrow
\text{aggregate}
\rightarrow
\text{update}
\rightarrow
\text{readout}.
$$

That grammar still appears inside modern graph transformers, equivariant GNNs, molecular encoders, and protein-ligand interaction models. Even when the architecture becomes more complex, the question remains:

$$
\text{what messages move between which entities, and how are they pooled?}
$$

## Links

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/molecular-modeling/index|Molecular modeling]]
- [[molecular-modeling/molecules|Molecules]]
- [[molecular-modeling/molecular-ligand|Molecular and ligand modeling]]
- [[molecular-modeling/data-evaluation|Data and Evaluation]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[papers/architectures/gcn|GCN]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/graph-isomorphism-network|GIN]]
- [[papers/architectures/graphormer|Graphormer]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]

## One-Line Memory

MPNN is the molecular graph paper that made message, update, and readout the standard architecture contract for graph neural networks.
