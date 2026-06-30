---
title: Deep Sets
aliases:
  - papers/deep-sets
tags:
  - papers
  - architectures
  - set-model
---

# Deep Sets

> The paper gives a simple architecture form for permutation-invariant functions over sets.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Deep Sets |
| Authors | Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander J. Smola |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1703.06114](https://arxiv.org/abs/1703.06114) |
| Status | verified |

## Question

Many inputs are sets: order should not change the output. The question was how to parameterize neural networks so that they respect permutation invariance rather than learning it accidentally from data.

## Main Claim

Permutation-invariant functions over sets can be represented with a sum-decomposition form:

$$
f(X)
=
\rho
\left(
\sum_{x \in X}
\phi(x)
\right)
$$

where $\phi$ embeds each element and $\rho$ maps the pooled set representation to the output.

## Method

The architecture separates element-wise processing from set-level aggregation.

| Component | Role |
| --- | --- |
| $\phi$ | maps each element independently |
| sum pooling | removes dependence on input order |
| $\rho$ | maps pooled representation to prediction |

For an equivariant set-to-set output, each element can also depend on a pooled global summary:

$$
y_i
=
\psi
\left(
x_i,
\sum_{x_j \in X}\phi(x_j)
\right)
$$

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Sum-decomposition is a principled invariant form | theoretical characterization and experiments | assumptions matter for exact representation |
| Set architectures help unordered-input tasks | set classification and related experiments | not all set interactions are captured equally well by simple pooling |
| Invariance should be built into architecture | comparison with order-sensitive baselines | later work adds attention for richer interactions |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | learning functions on sets |
| Input/output unit | unordered collection to set-level or element-level prediction |
| Architecture family | permutation-invariant/equivariant set networks |
| Not directly tested | high-cost pairwise interactions, long-context language modeling |

## Limitations

- Simple sum pooling can bottleneck complex pairwise or higher-order interactions.
- Invariance is correct only when the task truly ignores order.
- For geometric data, permutation invariance is not enough; rotation, translation, and reflection behavior may also matter.
- Later models such as Set Transformer add attention to improve interactions among elements.

## Why It Matters

Deep Sets is a clean anchor for deciding when the input object is a set rather than a sequence, graph, or grid.

## Connections

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[papers/architectures/index|Architecture papers]]
