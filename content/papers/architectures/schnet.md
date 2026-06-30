---
title: SchNet
aliases:
  - papers/schnet
  - papers/continuous-filter-convolution
  - papers/schnet-continuous-filter-convolutional-neural-network
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - molecular-modeling
---

# SchNet

> The paper introduced continuous-filter convolutions for atomistic systems with arbitrary 3D positions.

## Metadata

| Field | Value |
| --- | --- |
| Paper | SchNet: A continuous-filter convolutional neural network for modeling quantum interactions |
| Authors | Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1706.08566](https://arxiv.org/abs/1706.08566) |
| NeurIPS PDF | [paper 6700](https://papers.neurips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions.pdf) |
| Status | verified |

## Question

CNNs work well on grids because images, audio, and feature maps have regular neighborhoods. Molecules do not live on a fixed grid. Atoms have arbitrary 3D positions:

$$
\{(z_i,r_i)\}_{i=1}^{N},
\qquad
r_i\in\mathbb{R}^{3}.
$$

The architecture question is:

$$
\text{How can a neural network model local interactions between atoms without voxelizing space?}
$$

SchNet answers with continuous-filter convolution: filters are generated from interatomic distances instead of indexed by grid offsets.

## Main Claim

SchNet proposes a neural architecture for atomistic systems that uses continuous-filter convolutional layers over atoms in 3D space.

The core claim is:

$$
\text{atom embeddings}
+
\text{distance-conditioned filters}
+
\text{interaction blocks}
\Rightarrow
\text{learned molecular energy and forces}.
$$

The durable architecture idea is:

$$
\text{convolution on a grid}
\rightarrow
\text{continuous convolution over coordinates}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | atomic numbers $z_i$ and positions $r_i$ |
| Node unit | atom |
| Edge/relation | pairwise distance or nearby atom pair |
| Main operator | continuous-filter convolution |
| Filter input | interatomic distance $d_{ij}=\lVert r_i-r_j\rVert$ |
| Hidden state | atom-wise representation |
| Output | molecular property, especially total energy |
| Force route | negative gradient of energy with respect to coordinates |
| Symmetry target | permutation invariance, translation invariance, rotation-invariant energy |

SchNet is a molecular architecture note, not just a chemistry benchmark note. It defines how to mix atom features when atoms have continuous coordinates.

## From Grid Convolution to Continuous Filter

A standard discrete convolution over a grid is:

$$
y_i
=
\sum_{j}
x_j W_{i-j}.
$$

The filter $W_{i-j}$ depends on a discrete offset. That assumes grid positions.

For atoms, there is no fixed grid offset. SchNet replaces the discrete filter by a continuous function of relative geometry:

$$
x_i'
=
\sum_{j}
x_j \odot W(d_{ij}),
$$

where:

$$
d_{ij}
=
\lVert r_i-r_j\rVert_2.
$$

The filter-generating network maps distance to weights:

$$
W(d_{ij})
=
\operatorname{NN}_{\theta}(d_{ij}).
$$

This is the central architecture move:

$$
\text{filter lookup by grid offset}
\rightarrow
\text{filter generation by continuous distance}.
$$

## Why Distance?

If a molecular energy is invariant to global rotation and translation, the model should not change when the whole molecule is moved or rotated.

For translation $t$:

$$
r_i' = r_i + t,
$$

distances are unchanged:

$$
\lVert r_i'-r_j'\rVert
=
\lVert (r_i+t)-(r_j+t)\rVert
=
\lVert r_i-r_j\rVert.
$$

For rotation $R\in SO(3)$:

$$
r_i'=Rr_i,
$$

distances are also unchanged:

$$
\lVert Rr_i-Rr_j\rVert
=
\lVert R(r_i-r_j)\rVert
=
\lVert r_i-r_j\rVert.
$$

By using distances for filters, SchNet makes scalar energy prediction naturally invariant to global translation and rotation.

## Continuous-Filter Convolution

Let $h_i^{(t)}$ be the atom representation at interaction block $t$.

A simplified SchNet-style interaction update is:

$$
m_i^{(t)}
=
\sum_{j\in \mathcal{N}(i)}
h_j^{(t)}\odot W^{(t)}(d_{ij}),
$$

$$
h_i^{(t+1)}
=
h_i^{(t)}+\phi^{(t)}(m_i^{(t)}).
$$

Here:

| Symbol | Meaning |
| --- | --- |
| $h_i^{(t)}$ | atom embedding at layer $t$ |
| $\mathcal{N}(i)$ | atoms within a cutoff or selected neighborhood |
| $d_{ij}$ | interatomic distance |
| $W^{(t)}(d_{ij})$ | continuous filter generated from distance |
| $\odot$ | elementwise feature modulation |
| $\phi^{(t)}$ | learned atom-wise transformation |

This resembles message passing:

$$
m_{ij}
=
M(h_i,h_j,d_{ij}),
$$

but the defining feature is the continuous distance-conditioned filter.

## Radial Basis Expansion

Distances are continuous scalar inputs. A common SchNet-style implementation expands distances into radial basis features:

$$
e_k(d)
=
\exp\left(-\gamma(d-\mu_k)^2\right),
$$

where $\mu_k$ are radial centers.

Then the filter network consumes:

$$
e(d_{ij})
=
[e_1(d_{ij}),\ldots,e_K(d_{ij})].
$$

This makes it easier for the model to learn smooth functions over distance.

## Energy and Forces

SchNet predicts atom-wise contributions and sums them for total energy:

$$
E
=
\sum_i E_i.
$$

Forces can be obtained from the energy gradient:

$$
F_i
=
-\frac{\partial E}{\partial r_i}.
$$

This is important because it links the model to a smooth potential energy surface. If $E(r)$ is differentiable, force predictions are conservative by construction:

$$
\nabla \times F = 0
$$

under the energy-gradient formulation.

The architectural point is:

$$
\text{differentiable coordinate-dependent energy}
\rightarrow
\text{force from gradient}.
$$

## Relation to MPNN

[[papers/architectures/neural-message-passing-for-quantum-chemistry|MPNN]] gives the generic graph interface:

$$
m_i
=
\sum_{j\in N(i)}
M(h_i,h_j,e_{ij}).
$$

SchNet specializes this for atomistic geometry:

$$
e_{ij}
\rightarrow
d_{ij}
=
\lVert r_i-r_j\rVert.
$$

| Axis | MPNN | SchNet |
| --- | --- | --- |
| graph relation | generic edge feature | interatomic distance |
| domain | molecular graph property prediction | atomistic quantum interactions |
| geometry | optional | central |
| filter | message function | continuous distance-conditioned filter |
| output | graph property | energy and forces |

SchNet should be read after MPNN if the goal is molecular modeling.

## Relation to GNNs and Equivariance

SchNet's energy prediction is invariant because it uses distances and a permutation-invariant sum:

$$
E(G)
=
\sum_i E_i.
$$

It is not an equivariant vector-feature architecture in the later SE(3)-Transformer or tensor-field sense. Forces are equivariant because they are gradients of an invariant scalar energy:

$$
E(Rr)=E(r)
\Rightarrow
F(Rr)=RF(r).
$$

This distinction matters:

| Model Type | Scalar Energy | Vector Features |
| --- | --- | --- |
| SchNet | invariant by distance-based construction | forces via energy gradient |
| EGNN | coordinate updates can be equivariant | coordinate/message update |
| SE(3)-Transformer | tensor features transform under irreps | explicit equivariant attention |

SchNet is an important bridge from molecular GNNs to geometric deep learning.

## Relation to CNNs

SchNet extends the convolution idea:

$$
\text{local weighted aggregation}
$$

from grid data to atoms at arbitrary positions.

| Axis | CNN | SchNet |
| --- | --- |
| position domain | grid | continuous 3D coordinates |
| filter index | discrete offset | interatomic distance |
| translation sharing | grid offset sharing | distance-conditioned sharing |
| local neighborhood | kernel window | cutoff/radius neighborhood |
| output | image/audio/grid features | atom-wise and molecular properties |

This makes SchNet a molecular analogue of convolution, not merely a graph model with chemistry features.

## Evidence to Read

The paper's evidence should be read around atomistic modeling:

| Evidence | What It Supports |
| --- | --- |
| equilibrium molecule benchmarks | coordinate-aware representations improve molecular property prediction |
| molecular dynamics trajectories | smooth potential energy surfaces matter |
| energy and force prediction | gradient-based force route is physically meaningful |
| chemical/structural variation benchmark | generalization across chemistry and geometry is harder |
| comparison to hand-crafted descriptors | learned continuous filters can replace fixed molecular descriptors |

The main claim is strongest when both energy accuracy and force consistency are considered.

## Evaluation Risks

For atomistic models, ask:

| Risk | Check |
| --- | --- |
| conformer leakage | are near-identical structures in train and test? |
| molecule leakage | are same molecules split only by conformation? |
| unit mismatch | energy, force, and property units must be explicit |
| force consistency | are forces predicted directly or as energy gradients? |
| cutoff artifacts | does the radius cutoff miss long-range interactions? |
| target heterogeneity | different quantum properties require different physical context |

Architecture gains can be confused with better featurization, better conformer sampling, or easier splits.

## Failure Modes and Caveats

- Distances alone do not encode chirality in all settings.
- Long-range electrostatics may be hard with a short cutoff.
- Energy accuracy and force accuracy can trade off depending on training objective.
- SchNet predates many later equivariant architectures; it is geometry-aware but not a full tensor-feature equivariant model.
- Molecular benchmarks can be sensitive to split design and conformer overlap.

## Why This Matters for Architecture Reading

SchNet teaches a reusable lesson:

$$
\text{when the object is not on a grid, make the filter a function of geometry}.
$$

For structure-based AI, this is foundational. Protein-ligand models, atomistic force fields, molecular property predictors, and geometric GNNs all need to decide how coordinates enter the architecture.

The SchNet reading question is:

$$
\text{Are we modeling geometry as features, as filters, as coordinates, or as equivariant objects?}
$$

## Links

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[molecular-modeling/geometry|Geometry]]
- [[molecular-modeling/molecules|Molecules]]
- [[papers/architectures/neural-message-passing-for-quantum-chemistry|Neural Message Passing]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]

## One-Line Memory

SchNet is the atomistic architecture paper that replaces grid convolution with distance-conditioned continuous filters over atoms in 3D space.
