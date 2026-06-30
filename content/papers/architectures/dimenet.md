---
title: DimeNet
aliases:
  - papers/dimenet
  - papers/directional-message-passing
  - papers/directional-message-passing-for-molecular-graphs
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - molecular-modeling
---

# DimeNet

> The paper introduced directional message passing for molecular graphs, using distances and angles rather than only pairwise distances.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Directional Message Passing for Molecular Graphs |
| Authors | Johannes Gasteiger, Janek Gross, Stephan Gunnemann |
| Year | 2020 |
| Venue | ICLR 2020 |
| arXiv | [2003.03123](https://arxiv.org/abs/2003.03123) |
| OpenReview PDF | [B1eWbxStPH](https://openreview.net/pdf?id=B1eWbxStPH) |
| Project | [DimeNet project page](https://www.cs.cit.tum.de/en/daml/dimenet/) |
| Status | verified |

## Question

Many molecular GNNs use distances between atoms:

$$
d_{ij}
=
\lVert r_i-r_j\rVert.
$$

Distances are rotation and translation invariant, but they do not fully describe directional local geometry. Molecular interactions often depend on bond angles and angular potentials. DimeNet asks:

$$
\text{Can molecular message passing use directional information without breaking geometric symmetry?}
$$

The key move is to represent messages, not only atoms, and associate each message with a direction in coordinate space.

## Main Claim

DimeNet proposes directional message passing. Messages are attached to directed edges:

$$
m_{j\to i}
$$

rather than only node states:

$$
h_i.
$$

The update uses both distances and angles:

$$
d_{ij}
=
\lVert r_i-r_j\rVert,
\qquad
\theta_{kji}
=
\angle(r_k-r_j,\ r_i-r_j).
$$

The durable claim is:

$$
\text{distance-only molecular GNN}
\rightarrow
\text{directional message passing with angular information}.
$$

This makes DimeNet a key bridge between SchNet-style distance models and later geometric molecular architectures.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | atoms, coordinates, and molecular graph/radius graph |
| Node unit | atom |
| Message unit | directed edge/message $j\to i$ |
| Geometric inputs | interatomic distances and bond angles |
| Main operator | directional message passing |
| Basis functions | spherical Bessel functions and spherical harmonics |
| Output | molecular property, energy, force-related targets |
| Symmetry target | invariant scalar prediction with direction-aware intermediate messages |
| Main domain | molecular property prediction and quantum chemistry |

## Why Messages Instead of Only Atoms?

In a standard MPNN, a node aggregates messages from neighbors:

$$
m_i
=
\sum_{j\in N(i)}
M(h_i,h_j,e_{ij}).
$$

DimeNet treats the message $m_{j\to i}$ as the carrier of geometry. A message has an associated direction:

$$
e_{j\to i}
=
\frac{r_i-r_j}{\lVert r_i-r_j\rVert}.
$$

When the molecule rotates by $R$, the direction rotates:

$$
e_{j\to i}'
=
R e_{j\to i}.
$$

The architecture can use angles between directions, which are invariant to global rotation:

$$
\cos\theta_{kji}
=
\frac{(r_k-r_j)^\top(r_i-r_j)}
{\lVert r_k-r_j\rVert\lVert r_i-r_j\rVert}.
$$

This lets the model encode directional geometry while keeping final scalar predictions invariant.

## Directional Message Passing

Consider two directed messages:

$$
m_{k\to j}
\quad\text{and}\quad
m_{j\to i}.
$$

DimeNet transforms messages using the angle between their directions:

$$
\theta_{kji}
=
\angle(k\to j,\ j\to i).
$$

A simplified update is:

$$
m_{j\to i}^{(t+1)}
=
\sum_{k\in N(j)\setminus\{i\}}
\Phi
\left(
m_{k\to j}^{(t)},
d_{ji},
d_{kj},
\theta_{kji}
\right).
$$

The exclusion $k\ne i$ avoids immediate backtracking in the message update.

This resembles belief propagation:

$$
\text{incoming messages to }j
\rightarrow
\text{outgoing message }j\to i.
$$

The architecture is edge/message-centric, not only node-centric.

## Distances and Angles

Distance features encode pairwise separation:

$$
d_{ij}
=
\lVert r_i-r_j\rVert.
$$

Angular features encode triplet geometry:

$$
\theta_{kji}
=
\arccos
\left(
\frac{(r_k-r_j)^\top(r_i-r_j)}
{\lVert r_k-r_j\rVert\lVert r_i-r_j\rVert}
\right).
$$

The triplet $(k,j,i)$ is important because it describes how two interactions meet at atom $j$.

| Feature | Geometry |
| --- | --- |
| $d_{ij}$ | pairwise distance |
| $\theta_{kji}$ | local angle at atom $j$ |
| directed edge $j\to i$ | orientation of message flow |
| triplet $k\to j\to i$ | angular dependency between messages |

## Basis Functions

DimeNet uses physically motivated basis representations instead of only Gaussian radial basis functions.

For distances, it uses radial basis functions related to spherical Bessel functions:

$$
R_n(d)
$$

For angles, it uses spherical harmonics:

$$
Y_l^m(\theta,\varphi).
$$

In molecular graphs, the combined basis represents distance-angle structure:

$$
B_{n,l}(d,\theta)
\approx
R_n(d)Y_l(\theta).
$$

The reading point is not to memorize every basis detail. The important architecture idea is:

$$
\text{geometric feature}
\rightarrow
\text{orthogonal basis expansion}
\rightarrow
\text{message update}.
$$

This gives the network a structured way to model smooth molecular geometry.

## Relation to SchNet

[[papers/architectures/schnet|SchNet]] uses distance-conditioned filters:

$$
m_i
=
\sum_j h_j\odot W(d_{ij}).
$$

DimeNet adds directional message embeddings and angles:

$$
m_{j\to i}
\leftarrow
\Phi(m_{k\to j},d_{ji},d_{kj},\theta_{kji}).
$$

| Axis | SchNet | DimeNet |
| --- | --- | --- |
| primary geometry | distance | distance + angle |
| representation center | atom states | directed messages |
| interaction unit | pair $i,j$ | triplet $k,j,i$ |
| basis | radial distance features | radial + angular basis |
| physical motivation | continuous filter over atom distances | angular molecular interactions |

DimeNet should be read as a direct refinement of distance-only atomistic GNNs.

## Relation to MPNN

[[papers/architectures/neural-message-passing-for-quantum-chemistry|MPNN]] defines:

$$
m_i
=
\sum_{j\in N(i)}M(h_i,h_j,e_{ij}).
$$

DimeNet changes the unit being updated:

$$
h_i
\quad\rightarrow\quad
m_{j\to i}.
$$

This is a deeper change than adding one extra edge feature. It changes the computational graph from atom updates to directed-message updates.

## Relation to Equivariant GNNs

DimeNet uses directional information, but it is not the same as a full tensor-field equivariant network.

| Model | Geometry Handling |
| --- | --- |
| SchNet | invariant distance-based filters |
| DimeNet | directional messages and angular basis |
| EGNN | equivariant coordinate updates |
| SE(3)-Transformer | equivariant tensor features and attention |

DimeNet's final molecular property prediction is invariant, while the message directions rotate with the molecule. Later equivariant architectures make transformation behavior more explicit in feature types.

## Evidence to Read

DimeNet is evaluated on molecular property and force/energy style benchmarks.

| Evidence | What It Supports |
| --- | --- |
| QM9 | angular message passing improves equilibrium molecular property prediction |
| MD17 | direction-aware modeling helps molecular dynamics trajectories |
| comparison to SchNet-like baselines | angles and directed messages matter |
| parameter efficiency | orthogonal basis can improve performance with fewer parameters |
| ablations over geometric features | distance-only vs distance-angle reasoning |

The central evidence question is:

$$
\text{Does adding angular information improve molecular predictions beyond distance-only message passing?}
$$

## Evaluation Risks

For DimeNet-style models, check:

| Risk | Check |
| --- | --- |
| conformer availability | true 3D coordinates may not be available in real screening tasks |
| conformer leakage | near-identical geometries across splits can inflate results |
| computational cost | triplet interactions can be more expensive than pairwise messages |
| cutoff sensitivity | angular triplets depend on radius graph construction |
| target type | some targets need long-range electrostatics beyond local angles |
| coordinate quality | noisy conformers can degrade angle-based features |

Adding angles is not automatically better if the geometry is unreliable or unavailable.

## Failure Modes and Caveats

- Directional message passing requires 3D coordinates or plausible synthetic coordinates.
- Triplet construction can increase cost significantly.
- Angular features may overfit conformer-specific artifacts if splits are weak.
- Distance and angle features are invariant summaries; they do not expose the full vector/tensor representation used by later equivariant networks.
- Molecular long-range effects may require explicit global or electrostatic modeling.

## Why This Matters for Architecture Reading

DimeNet teaches a key geometric modeling lesson:

$$
\text{pairwise distances}
\neq
\text{full local molecular geometry}.
$$

Angles are part of molecular structure. For protein-ligand and molecular modeling work, this paper is a useful checkpoint before reading more complex equivariant architectures.

The reading question is:

$$
\text{Does the architecture use enough geometry for the target, without assuming unavailable coordinates?}
$$

## Links

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[molecular-modeling/geometry|Geometry]]
- [[papers/architectures/neural-message-passing-for-quantum-chemistry|Neural Message Passing]]
- [[papers/architectures/schnet|SchNet]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]

## One-Line Memory

DimeNet is the molecular GNN paper that moves from distance-only atom interactions to directed messages transformed by distances and angles.
