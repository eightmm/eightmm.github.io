---
title: Tensor Field Networks
aliases:
  - papers/tensor-field-networks
  - papers/tfn
  - papers/rotation-translation-equivariant-neural-networks-3d-point-clouds
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - equivariance
---

# Tensor Field Networks

> The paper introduced neural layers for 3D point clouds that are equivariant to rotations, translations, and permutations by using tensor features and spherical-harmonic filters.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds |
| Authors | Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, Patrick Riley |
| Year | 2018 |
| Venue | arXiv |
| arXiv | [1802.08219](https://arxiv.org/abs/1802.08219) |
| Code | [tensorfieldnetworks/tensorfieldnetworks](https://github.com/tensorfieldnetworks/tensorfieldnetworks) |
| Status | full note started |

## One-Line Takeaway

Tensor Field Networks make equivariance a layer-level contract: each feature channel has a rotation type, and filters built from spherical harmonics guarantee that outputs transform with the correct type.

## Question

3D point clouds and molecular structures can be rotated or translated without changing their intrinsic identity:

$$
x_i'
=
Rx_i+t,
\qquad
R\in SO(3),
\quad
t\in\mathbb{R}^3.
$$

For scalar predictions, the output should be invariant:

$$
f(\{Rx_i+t\})
=
f(\{x_i\}).
$$

For vector or tensor predictions, the output should transform predictably:

$$
F(\{Rx_i+t\})
=
\rho(R)F(\{x_i\}).
$$

The paper asks:

$$
\text{Can every hidden layer of a point-cloud network preserve this geometric transformation law?}
$$

TFN answers with tensor-valued features and equivariant convolutional filters.

## Main Claim

Tensor field networks define convolution-like message passing over 3D points such that each layer is equivariant to:

$$
\text{translation}
\times
\text{rotation}
\times
\text{point permutation}.
$$

The core architecture pattern is:

$$
\text{typed tensor features}
+
\text{spherical-harmonic geometric filters}
+
\text{symmetry-compatible mixing}
\rightarrow
\text{typed tensor features}.
$$

This is a foundation for later equivariant geometric architectures such as [[papers/architectures/se3-transformer|SE(3)-Transformer]], [[papers/architectures/egnn|EGNN]], and molecular models like [[papers/architectures/painn|PaiNN]].

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | point coordinates $x_i\in\mathbb{R}^3$ and point features |
| Unit | point, atom, residue, or node with 3D coordinates |
| Hidden state | tensor features separated by rotation type $\ell$ |
| Geometry | relative displacement $r_{ij}=x_j-x_i$ |
| Filter | radial functions times spherical harmonics |
| Main operator | equivariant convolution/message passing |
| Symmetry | translation equivariance, rotation equivariance, permutation equivariance |
| Output | scalar, vector, or higher-order tensor features |

The important contract is not only final invariance. Every intermediate feature has a declared transformation rule.

## Translation And Permutation

Translation is handled by using relative positions:

$$
r_{ij}
=
x_j-x_i.
$$

Under a global translation $t$:

$$
r_{ij}'
=
(x_j+t)-(x_i+t)
=
r_{ij}.
$$

Permutation equivariance comes from shared local operations and summation over neighbors:

$$
h_i'
=
\sum_{j\in\mathcal{N}(i)}
m(h_j,r_{ij}).
$$

If point order changes, the same set of messages is summed in a different order, producing the correspondingly permuted output.

## Feature Types

TFN organizes hidden features by rotation order $\ell$.

| Type | Meaning | Transform Rule |
| --- | --- | --- |
| $\ell=0$ | scalar | unchanged under rotation |
| $\ell=1$ | vector-like feature | rotates with $R$ |
| $\ell=2$ | rank-2 tensor-like feature | transforms with $D^{(2)}(R)$ |
| $\ell>2$ | higher-order angular feature | transforms with $D^{(\ell)}(R)$ |

A feature of type $\ell$ transforms as:

$$
f_i^{(\ell)}
\mapsto
D^{(\ell)}(R)f_i^{(\ell)},
$$

where $D^{(\ell)}(R)$ is the Wigner-D representation for rotation $R$.

This is the key idea:

$$
\text{feature channel}
\neq
\text{ordinary vector of numbers only};
$$

it also has a geometric transformation type.

## Spherical Harmonic Filters

The relative displacement is:

$$
r_{ij}
=
x_j-x_i.
$$

Write it in radial and angular components:

$$
r_{ij}
\leftrightarrow
(\lVert r_{ij}\rVert,\hat{r}_{ij}).
$$

TFN uses filters of the form:

$$
F_{\ell m}(r_{ij})
=
R_{\ell}(\lVert r_{ij}\rVert)
Y_{\ell m}(\hat{r}_{ij}),
$$

where:

| Symbol | Meaning |
| --- | --- |
| $R_{\ell}$ | learnable radial function |
| $Y_{\ell m}$ | spherical harmonic basis |
| $\ell$ | angular frequency / representation order |
| $m$ | component index within order $\ell$ |

Spherical harmonics have the correct rotation behavior. Under rotation:

$$
Y_{\ell}(R\hat{r})
=
D^{(\ell)}(R)Y_{\ell}(\hat{r}).
$$

That is why they are useful as geometric filters.

## Equivariant Convolution

A simplified TFN layer maps input type $\ell_{\mathrm{in}}$ to output type $\ell_{\mathrm{out}}$ by combining input features with a filter type $\ell_f$:

$$
h_i^{(\ell_{\mathrm{out}})}
=
\sum_{j\in\mathcal{N}(i)}
\sum_{\ell_{\mathrm{in}},\ell_f}
C_{\ell_{\mathrm{in}},\ell_f\rightarrow\ell_{\mathrm{out}}}
\left(
h_j^{(\ell_{\mathrm{in}})}
\otimes
F^{(\ell_f)}(r_{ij})
\right).
$$

Here $C_{\ell_{\mathrm{in}},\ell_f\rightarrow\ell_{\mathrm{out}}}$ represents symmetry-compatible coupling, often described using Clebsch-Gordan-style tensor product rules.

The type-selection rule is:

$$
\ell_{\mathrm{out}}
\in
\{|\ell_{\mathrm{in}}-\ell_f|,\ldots,\ell_{\mathrm{in}}+\ell_f\}.
$$

So not every feature can mix with every filter to produce every output type. Representation theory constrains the architecture.

## Why Equivariance Holds

Suppose input features transform as:

$$
h_j^{(\ell_{\mathrm{in}})}
\mapsto
D^{(\ell_{\mathrm{in}})}(R)h_j^{(\ell_{\mathrm{in}})}.
$$

The filter transforms as:

$$
F^{(\ell_f)}(Rr_{ij})
=
D^{(\ell_f)}(R)F^{(\ell_f)}(r_{ij}).
$$

Their tensor product transforms as the product representation:

$$
D^{(\ell_{\mathrm{in}})}(R)
\otimes
D^{(\ell_f)}(R).
$$

Clebsch-Gordan decomposition selects components that transform as $\ell_{\mathrm{out}}$:

$$
D^{(\ell_{\mathrm{in}})}\otimes D^{(\ell_f)}
\rightarrow
D^{(\ell_{\mathrm{out}})}.
$$

Therefore:

$$
h_i^{(\ell_{\mathrm{out}})}
\mapsto
D^{(\ell_{\mathrm{out}})}(R)
h_i^{(\ell_{\mathrm{out}})}.
$$

That is the layer-level equivariance guarantee.

## Relation To Later Architectures

| Paper | What It Takes From This Lineage | Main Difference |
| --- | --- | --- |
| [SE(3)-Transformer](/papers/architectures/se3-transformer) | typed equivariant features and spherical harmonic geometry | adds attention-style weighting |
| [PaiNN](/papers/architectures/painn) | scalar/vector equivariant feature separation | restricts mostly to scalar and vector channels for atomistic efficiency |
| [EGNN](/papers/architectures/egnn) | coordinate-aware equivariance goal | avoids high-order tensor features and spherical harmonics |
| [DimeNet](/papers/architectures/dimenet) | angular geometry matters | uses invariant distance/angle basis on directed messages |
| [SchNet](/papers/architectures/schnet) | continuous geometry over atoms | uses distance-conditioned invariant filters |

TFN is heavier than many later models, but it explains the representation-theoretic foundation behind typed equivariant channels.

## Evidence To Read

| Evidence | What It Supports |
| --- | --- |
| point cloud tasks | rotation/translation equivariance is useful beyond grid data |
| physical and chemical examples | tensor features can represent direction-sensitive quantities |
| layer construction | equivariance is guaranteed by filter and coupling design, not learned by data augmentation |
| comparison with non-equivariant baselines | symmetry can reduce the need to see many rotated examples |

The paper is most valuable as an architecture reference. Its lasting contribution is the construction, not a single benchmark leaderboard result.

## Why This Matters

Tensor Field Networks make three ideas explicit:

1. hidden channels can be scalars, vectors, or higher-order tensors;
2. filters over 3D directions should transform according to spherical harmonics;
3. neural layers can be constrained so equivariance holds exactly.

This gives a rigorous language for reading geometric deep learning papers. When a newer model claims to be "SE(3)-equivariant" or "E(3)-equivariant," TFN explains what that claim means at the feature and layer level.

## Limitations

| Limitation | Why It Matters |
| --- | --- |
| Computational cost | higher-order features and tensor products can be expensive |
| Implementation complexity | correct irreps, spherical harmonics, and coupling rules are easy to get wrong |
| Nonlinearity design | nonlinear layers must preserve feature types |
| Practical scope | many tasks do not need high-order tensor channels |
| Benchmark translation | exact equivariance helps only when the task's symmetry assumptions match the data |

For many molecular property tasks, later architectures trade some generality for speed and simplicity. That is why [[papers/architectures/painn|PaiNN]] and [[papers/architectures/egnn|EGNN]] are useful follow-up readings.

## What To Remember

The compact memory is:

$$
\text{TFN}
=
\text{typed tensor features}
+
\text{spherical harmonic filters}
+
\text{equivariant tensor products}.
$$

Read it as the paper that turns "respect 3D rotations" into a concrete neural layer design.

## Links

- [[concepts/geometric-deep-learning/tensor-field-network|Tensor Field Network]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]
- [[papers/architectures/painn|PaiNN]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]
- [[papers/architectures/dimenet|DimeNet]]
