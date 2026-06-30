---
title: Weisfeiler-Lehman Test
aliases:
  - concepts/architectures/weisfeiler-lehman-test
  - concepts/architectures/1-wl
  - concepts/architectures/color-refinement
tags:
  - architectures
  - graph-neural-networks
  - graph-theory
---

# Weisfeiler-Lehman Test

The Weisfeiler-Lehman test is a graph-color refinement procedure used as a reference point for the expressive power of message-passing graph neural networks.

For the common 1-dimensional WL test, each node starts with an initial color:

$$
c_i^{(0)}
=
\operatorname{hash}(x_i)
$$

At iteration $t$, the node color is updated from its previous color and the multiset of neighbor colors:

$$
c_i^{(t+1)}
=
\operatorname{hash}
\left(
c_i^{(t)},
\{\!\{c_j^{(t)} : j \in \mathcal{N}(i)\}\!\}
\right).
$$

Two graphs are distinguishable by 1-WL if their color histograms differ at some iteration.

## Why It Matters For GNNs

Many message-passing GNNs have the same recursive shape:

$$
h_i^{(t+1)}
=
\phi
\left(
h_i^{(t)},
\operatorname{AGG}
\left(
\{\!\{h_j^{(t)} : j \in \mathcal{N}(i)\}\!\}
\right)
\right).
$$

This makes 1-WL a useful ceiling for standard message passing. If the aggregation/update cannot distinguish two WL color refinement states, the GNN usually cannot separate the corresponding graphs either.

The key design question is:

$$
\text{Is the neighborhood aggregation injective over multisets?}
$$

If aggregation collapses distinct multisets into the same vector, expressive power is lost before the learned MLP even sees the neighborhood.

## Multiset Aggregation

A set ignores multiplicity. A multiset preserves how many times an element appears:

$$
\{a,b\}
\neq
\{\!\{a,a,b\}\!\}.
$$

For graph neighborhoods, multiplicity matters because repeated neighbor labels encode degree and local structure.

Common aggregators behave differently:

| Aggregator | Multiset Sensitivity | Typical Issue |
| --- | --- | --- |
| sum | can preserve counts when followed by expressive maps | scale grows with degree |
| mean | normalizes degree away | different counts can collapse |
| max | keeps strongest feature only | repeated or weaker evidence disappears |

Example:

$$
\operatorname{mean}(\{1,1,3\})
=
\operatorname{mean}(\{1,3,3\})
=
\frac{5}{3},
$$

but the multisets are different.

## Connection To GIN

[[papers/architectures/graph-isomorphism-network|Graph Isomorphism Network]] uses sum aggregation and an MLP update to approximate an injective multiset function:

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

The paper's durable claim is not that every graph problem needs maximum WL power. The useful reading is:

$$
\text{aggregation choice}
\Rightarrow
\text{expressive ceiling}.
$$

## Limits

1-WL is not a complete graph isomorphism test. Some non-isomorphic graphs remain indistinguishable.

For ML reading, this means:

- matching 1-WL is a useful baseline for message-passing expressivity;
- exceeding 1-WL usually requires higher-order tensors, subgraph features, positional encodings, random features, structural encodings, or non-local attention;
- higher expressivity can increase compute, overfitting risk, and leakage risk if graph construction uses hidden target information.

## Reading Checks

| Question | Why |
| --- | --- |
| What is the node color or feature at initialization? | WL power depends on initial labels/features |
| Is aggregation injective over multisets? | non-injective aggregation collapses structures |
| Does the readout preserve graph-level multiplicity? | graph classification also needs invariant but expressive pooling |
| Does the task require structure beyond 1-WL? | cycles, motifs, and long-range patterns may need stronger bias |
| Is stronger expressivity actually measured? | benchmark gains may come from features, splits, or training budget |

## Related

- [[concepts/architectures/gnn|Graph Neural Networks]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[papers/architectures/graph-isomorphism-network|How Powerful are Graph Neural Networks?]]
