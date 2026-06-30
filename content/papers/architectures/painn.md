---
title: Equivariant Message Passing for Tensorial Molecular Properties
aliases:
  - papers/painn
  - papers/equivariant-message-passing-for-tensorial-properties
  - papers/polarizable-atom-interaction-neural-network
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - molecular-modeling
---

# Equivariant Message Passing for Tensorial Molecular Properties

> The paper introduced PaiNN, a molecular message-passing architecture that maintains scalar and vector atom features so it can predict rotationally equivariant tensorial properties.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Equivariant message passing for the prediction of tensorial properties and molecular spectra |
| Authors | Kristof T. Schütt, Oliver T. Unke, Michael Gastegger |
| Year | 2021 |
| Venue | ICML 2021 |
| arXiv | [2102.03150](https://arxiv.org/abs/2102.03150) |
| PMLR PDF | [v139/schutt21a](https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf) |
| Status | full note started |

## One-Line Takeaway

PaiNN keeps both invariant scalar features and equivariant vector features for each atom, letting molecular message passing propagate directional information without using heavy tensor product machinery at every layer.

## Question

Distance-based molecular models such as [[papers/architectures/schnet|SchNet]] are naturally invariant to translation and rotation:

$$
d_{ij}
=
\lVert r_i-r_j\rVert.
$$

This is good for scalar energy prediction, but some molecular properties are vectors or tensors. Examples include dipole moments, polarizability, and direction-sensitive spectral quantities.

The architecture question is:

$$
\text{How can a molecular GNN represent directional information while preserving E(3) symmetry?}
$$

PaiNN answers by giving each atom two feature channels:

$$
s_i \in \mathbb{R}^{F},
\qquad
V_i \in \mathbb{R}^{3\times F}.
$$

The scalar feature $s_i$ is invariant. The vector feature $V_i$ rotates with the molecule.

## Main Claim

Rotationally equivariant atom-wise vector representations can improve molecular property prediction and enable tensorial property prediction.

The durable architecture pattern is:

$$
\text{atom scalar state}
+
\text{atom vector state}
+
\text{equivariant message passing}
\rightarrow
\text{scalar and tensor molecular properties}.
$$

Compared with distance-only message passing:

$$
h_i
\rightarrow
h_i',
$$

PaiNN updates coupled scalar and vector states:

$$
(s_i,V_i)
\rightarrow
(s_i',V_i').
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | atomic numbers $z_i$ and coordinates $r_i\in\mathbb{R}^3$ |
| Node unit | atom |
| Edge geometry | relative direction $r_{ij}=r_j-r_i$ and distance $\lVert r_{ij}\rVert$ |
| Hidden state | scalar atom features $s_i$ and vector atom features $V_i$ |
| Main operator | equivariant message passing with scalar-vector coupling |
| Output | scalar molecular properties, vector properties, tensorial properties, spectra-related quantities |
| Symmetry target | translation invariance, rotational invariance for scalar outputs, rotational equivariance for vector/tensor outputs |
| Domain | molecular property prediction and molecular spectra |

The contract differs from invariant molecular GNNs:

$$
\text{scalar-only hidden state}
\quad\Rightarrow\quad
\text{good for invariant scalars}
$$

$$
\text{scalar + vector hidden state}
\quad\Rightarrow\quad
\text{can carry directional information}.
$$

## E(3) Symmetry

Let a global Euclidean transformation be:

$$
r_i'
=
Rr_i+t,
\qquad
R\in O(3),
\quad
t\in\mathbb{R}^{3}.
$$

Scalar atom features should stay unchanged:

$$
s_i'
=
s_i.
$$

Vector atom features should rotate:

$$
V_i'
=
RV_i.
$$

This is the central equivariance condition. If the molecule rotates, the internal vector features rotate in the same way. A scalar energy prediction should remain invariant:

$$
E(\{Rr_i+t\})
=
E(\{r_i\}),
$$

while a vector property should transform equivariantly:

$$
\mu(\{Rr_i+t\})
=
R\mu(\{r_i\}).
$$

## Scalar and Vector Features

PaiNN represents each atom with:

$$
s_i \in \mathbb{R}^{F},
\qquad
V_i =
\begin{bmatrix}
v_{i,1} & v_{i,2} & \cdots & v_{i,F}
\end{bmatrix}
\in
\mathbb{R}^{3\times F}.
$$

Each vector channel $v_{i,f}\in\mathbb{R}^3$ can rotate with the molecule. The scalar channel can store invariant chemical context, while the vector channel stores orientation-aware information.

This is less expressive than a full hierarchy of high-order irreducible representations, but it is easier to implement and efficient for atomistic prediction.

## Message Block

For an edge $j\to i$, define:

$$
r_{ij}=r_j-r_i,
\qquad
d_{ij}=\lVert r_{ij}\rVert,
\qquad
\hat{r}_{ij}=\frac{r_{ij}}{d_{ij}}.
$$

Radial functions use the distance:

$$
W(d_{ij})
=
\operatorname{MLP}_{\theta}
\left(
\operatorname{RBF}(d_{ij})
\right).
$$

A simplified scalar message has the form:

$$
\Delta s_i
=
\sum_{j\in\mathcal{N}(i)}
\phi_s(s_j,d_{ij}).
$$

The vector message can use both the neighbor vector feature and the normalized direction:

$$
\Delta V_i
=
\sum_{j\in\mathcal{N}(i)}
\left[
\phi_{vv}(s_j,d_{ij}) \odot V_j
+
\phi_{vs}(s_j,d_{ij}) \odot \hat{r}_{ij}
\right].
$$

This equation captures the architecture idea without tying the note to one implementation detail:

- scalar gates are invariant because they depend on scalar features and distances;
- $V_j$ rotates equivariantly;
- $\hat{r}_{ij}$ rotates equivariantly;
- sums of equivariant vectors remain equivariant.

Therefore:

$$
\Delta V_i'
=
R\Delta V_i.
$$

## Update Block

The update block mixes scalar and vector channels while preserving symmetry.

A common pattern is to derive scalar information from vector norms or dot products:

$$
\lVert V_i \rVert
=
\left(
\lVert v_{i,1}\rVert,\ldots,\lVert v_{i,F}\rVert
\right),
$$

then use scalar gates to modulate vector channels:

$$
V_i'
=
V_i
+
\gamma(s_i,\lVert V_i\rVert)\odot V_i.
$$

Scalar updates can depend on invariant vector summaries:

$$
s_i'
=
s_i
+
\psi(s_i,\lVert V_i\rVert).
$$

The design rule is:

$$
\text{scalar output}
\leftarrow
\text{invariant combinations},
$$

$$
\text{vector output}
\leftarrow
\text{scalar gates}\times\text{equivariant vectors}.
$$

This lets the model exchange information between scalar and directional channels without breaking equivariance.

## Relation To SchNet And DimeNet

| Paper | Geometry Carrier | Main Limitation Addressed By PaiNN |
| --- | --- | --- |
| [SchNet](/papers/architectures/schnet) | pairwise distances | distance-only invariant hidden states do not carry vector direction directly |
| [DimeNet](/papers/architectures/dimenet) | directed messages, distances, angles | direction is encoded through angular invariant features rather than persistent vector atom states |
| [PaiNN](/papers/architectures/painn) | scalar and vector atom features | direction is explicitly carried as equivariant vector channels |
| [EGNN](/papers/architectures/egnn) | coordinate updates | learns equivariant coordinate dynamics but has a different scalar-coordinate update contract |
| [SE(3)-Transformer](/papers/architectures/se3-transformer) | equivariant attention with representation structure | more general equivariant machinery, usually heavier |

PaiNN is useful to read after SchNet and DimeNet because it makes the hidden representation itself equivariant, not only the geometric input features.

## Tensorial Properties

For a vector molecular property such as dipole moment, the output must rotate with the molecule:

$$
\mu'
=
R\mu.
$$

An atom-wise vector readout can be written as:

$$
\mu
=
\sum_i
\mu_i(V_i,s_i,r_i).
$$

For rank-2 tensor properties such as polarizability, a simplified equivariant construction can combine invariant scalar coefficients with equivariant vector outer products:

$$
T
=
\sum_i
\sum_k
\lambda_{ik}
u_{ik}u_{ik}^{\top},
$$

where $u_{ik}$ is an equivariant vector derived from $V_i$ or coordinate-dependent vector features, and $\lambda_{ik}$ is invariant.

If coordinates rotate:

$$
u_{ik}'=Ru_{ik},
$$

then:

$$
T'
=
\sum_i\sum_k
\lambda_{ik}
(Ru_{ik})(Ru_{ik})^\top
=
RT R^\top.
$$

This is why tensorial molecular prediction is a stronger test than scalar energy alone.

## Evidence To Read

| Evidence | What It Supports |
| --- | --- |
| QM9 molecular property results | scalar molecular property prediction benefits from equivariant vector features |
| MD17 energy and force results | equivariant message passing can support molecular dynamics-style targets |
| tensorial property prediction | vector channels enable direction-sensitive outputs |
| spectra simulation case | learned potentials/tensorial properties can accelerate expensive electronic-structure workflows |
| ablations on equivariant features | vector channels are not only architectural decoration |

The paper reports that PaiNN improves common molecular benchmarks while reducing model size and inference time relative to some previous networks, and applies the architecture to molecular spectra simulation.

## Why This Matters

PaiNN is a central architecture for reading modern molecular geometric models because it separates three ideas cleanly:

1. invariant scalar chemical context;
2. equivariant vector directional context;
3. symmetry-preserving coupling between them.

This makes it a practical middle point between simple invariant GNNs and more general equivariant tensor/representation architectures.

## Limitations

| Limitation | Why It Matters |
| --- | --- |
| Mostly vector features | higher-order tensor information may require richer irreducible representations or tensor channels |
| Local cutoff dependence | long-range electrostatics and global conformational effects may be missed without additional mechanisms |
| Benchmark sensitivity | molecular benchmarks can be sensitive to split, conformer, and target preprocessing |
| Force consistency | if forces are derived from energy gradients, the energy architecture and training target must be handled carefully |
| Geometry availability | the model assumes 3D coordinates are available or generated elsewhere |
| Domain scope | atomistic property prediction differs from protein-ligand docking or sequence-only protein modeling |

For docking or protein-ligand modeling, PaiNN-style equivariant representations are useful building blocks, but the full task also needs conformer generation, pocket context, interaction labels, and domain-specific evaluation.

## What To Remember

The core memory is:

$$
\text{PaiNN}
=
\text{scalar atom features}
+
\text{vector atom features}
+
\text{equivariant message passing}.
$$

Read it as the point where molecular message passing stops treating geometry only as invariant edge features and starts carrying direction as part of the hidden state.

## Links

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]
- [[molecular-modeling/geometry-for-structure-modeling|Geometry for Structure Modeling]]
- [[papers/architectures/schnet|SchNet]]
- [[papers/architectures/dimenet|DimeNet]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]
