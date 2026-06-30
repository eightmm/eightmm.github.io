---
title: NequIP
aliases:
  - papers/nequip
  - papers/e3-equivariant-interatomic-potentials
  - papers/neural-equivariant-interatomic-potentials
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - equivariance
  - molecular-modeling
  - interatomic-potentials
---

# NequIP

> NequIP is an E(3)-equivariant graph neural network for learning interatomic potentials with tensor features that rotate consistently with atomic geometry.

## Metadata

| Field | Value |
| --- | --- |
| Paper | E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials |
| Authors | Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt, Boris Kozinsky |
| Year | 2022 |
| Venue | Nature Communications |
| arXiv | [2101.03164](https://arxiv.org/abs/2101.03164) |
| DOI | [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5) |
| Status | full note started |

## One-Line Takeaway

NequIP makes molecular dynamics potentials a geometric architecture problem: if the target energy and forces obey Euclidean symmetry, the network should preserve that symmetry in every layer rather than learn it from data.

## Question

An atomistic configuration is a set of atom types and coordinates:

$$
\mathcal{X}
=
\{(z_i,r_i)\}_{i=1}^{N},
\qquad
r_i\in\mathbb{R}^3.
$$

For an interatomic potential, the scalar energy should be invariant to rigid motion:

$$
E(\{Rr_i+t\})
=
E(\{r_i\}),
\qquad
R\in O(3).
$$

The force on atom $i$ is the negative energy gradient:

$$
F_i
=
-\nabla_{r_i}E.
$$

Forces should rotate with the system:

$$
F_i(\{Rr_j+t\})
=
R F_i(\{r_j\}).
$$

The architecture question is:

$$
\text{Can a neural interatomic potential build this symmetry into the representation itself?}
$$

NequIP answers with E(3)-equivariant message passing over atoms using tensor features and equivariant convolutions.

## Main Claim

The durable claim is not just "better molecular property prediction." It is:

$$
\text{equivariant tensor features}
+ \text{local atomic graph}
+ \text{energy-force training}
\rightarrow
\text{data-efficient interatomic potentials}.
$$

Compared with scalar invariant molecular GNNs, NequIP keeps richer hidden features that transform under rotations. This lets the model represent directional atomic environments without requiring the training data to teach rotational physics from scratch.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | atom types $z_i$ and coordinates $r_i$ |
| Node unit | atom |
| Edge geometry | relative vector $r_{ij}=r_j-r_i$, distance $\lVert r_{ij}\rVert$, and angular basis |
| Graph | local neighbor graph under a cutoff radius |
| Hidden state | typed tensor features indexed by irreducible representation order $\ell$ |
| Main operator | E(3)-equivariant convolution/message passing |
| Output | atom-wise energy contributions and total energy |
| Force prediction | differentiable force from energy gradient |
| Symmetry target | translation invariance, rotation/reflection equivariance, permutation equivariance |
| Domain | molecular and materials interatomic potentials |

The output contract usually has:

$$
E_\theta(\mathcal{X})
=
\sum_i E_{\theta,i},
$$

then:

$$
F_{\theta,i}
=
-\frac{\partial E_\theta}{\partial r_i}.
$$

Predicting forces as energy gradients enforces a conservative force field, assuming the energy model is differentiable and trained appropriately.

## E(3) Equivariance

E(3) includes translation, rotation, and reflection. The input coordinates transform as:

$$
r_i'
=
Rr_i+t,
\qquad
R\in O(3).
$$

Translation is handled with relative coordinates:

$$
r_{ij}
=
r_j-r_i.
$$

Under translation:

$$
(r_j+t)-(r_i+t)=r_{ij}.
$$

Under rotation or reflection:

$$
r_{ij}'
=
Rr_{ij}.
$$

An equivariant layer preserves the transformation rule:

$$
h_i'^{(\ell)}
=
D^{(\ell)}(R)h_i^{(\ell)},
$$

where $h_i^{(\ell)}$ is a feature of order $\ell$ and $D^{(\ell)}(R)$ is the corresponding representation matrix.

This is the same broad design family as [[papers/architectures/tensor-field-networks|Tensor Field Networks]] and [[papers/architectures/se3-transformer|SE(3)-Transformer]], but specialized into an effective atomistic potential architecture.

## Tensor Features

Scalar-only GNNs store features like:

$$
h_i\in\mathbb{R}^{F}.
$$

These features are invariant to global rotation. That is useful for scalar readouts, but it can discard directional information.

NequIP stores channels by geometric type:

| Type | Meaning | Transform Rule |
| --- | --- | --- |
| $\ell=0$ | scalar feature | unchanged |
| $\ell=1$ | vector-like feature | rotates like a vector |
| $\ell=2$ | higher angular feature | rotates by $D^{(2)}(R)$ |
| $\ell>2$ | higher-order tensor feature | rotates by $D^{(\ell)}(R)$ |

The useful mental model is:

$$
\text{feature value}
+ \text{feature type}
=
\text{geometric representation}.
$$

The type tells the network how that channel must transform when the molecule is rotated.

## Equivariant Convolution Sketch

For an edge $j\to i$, use:

$$
r_{ij}=r_j-r_i,
\qquad
d_{ij}=\lVert r_{ij}\rVert,
\qquad
\hat{r}_{ij}=r_{ij}/d_{ij}.
$$

The angular part is represented by spherical harmonics:

$$
Y_{\ell m}(\hat{r}_{ij}).
$$

The radial part is learned from distances:

$$
\phi_{\theta}(d_{ij})
=
\operatorname{MLP}_{\theta}
\left(
\operatorname{RBF}(d_{ij})
\right).
$$

An equivariant message can be read abstractly as:

$$
m_{ij}
=
\operatorname{TensorProduct}
\left(
h_j,\,
\phi_{\theta}(d_{ij})Y(\hat{r}_{ij})
\right).
$$

The node update sums neighbor messages:

$$
h_i'
=
\sum_{j\in\mathcal{N}(i)} m_{ij}.
$$

The exact implementation uses irreducible representation bookkeeping and parity-aware tensor products. The architecture-level point is simpler:

$$
\text{distance controls radial strength},
\qquad
\text{direction controls angular transformation}.
$$

## Energy and Force Training

Interatomic potentials are often trained against both energies and forces:

$$
\mathcal{L}
=
\lambda_E
\lVert E_\theta-E^\star\rVert^2
+
\lambda_F
\sum_i
\lVert F_{\theta,i}-F_i^\star\rVert^2.
$$

Because:

$$
F_{\theta,i}
=
-\nabla_{r_i}E_\theta,
$$

force supervision provides dense geometric learning signal. For $N$ atoms, there are $3N$ force components but only one total energy. This helps explain why force labels are central in machine-learned potentials.

## Why It Matters Architecturally

NequIP is important because it turns a domain law into a model contract.

| Design Choice | Why It Matters |
| --- | --- |
| local atomic graph | gives a scalable inductive bias for atomistic interactions |
| E(3)-equivariant hidden state | preserves rotational/reflection behavior through layers |
| tensor products | mixes geometry and learned features without breaking equivariance |
| energy decomposition | maps local messages to a global scalar potential |
| force from gradient | makes force prediction consistent with the learned energy |

This is a stronger claim than "use a GNN on molecules." The claim is that geometry-aware representation type is part of the architecture.

## Relation to Other Architecture Notes

| Paper | Relation |
| --- | --- |
| [[papers/architectures/schnet|SchNet]] | distance-based atomistic network; invariant scalar features |
| [[papers/architectures/tensor-field-networks|Tensor Field Networks]] | earlier tensor-feature equivariant layers |
| [[papers/architectures/se3-transformer|SE(3)-Transformer]] | equivariant attention over 3D structures |
| [[papers/architectures/painn|PaiNN]] | scalar/vector molecular features for tensorial properties |
| [[papers/architectures/egnn|EGNN]] | simpler coordinate-update equivariant GNN |
| [[papers/architectures/dimenet|DimeNet]] | molecular directional messages with angular basis functions |

NequIP is closest to TFN-style typed tensor features, but it is framed around practical energy and force prediction for molecular dynamics and materials simulation.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| equivariance improves data efficiency | benchmark comparisons across molecules/materials | geometric symmetry can reduce sample demand | depends on benchmark, training labels, and baseline tuning |
| tensor features help atomistic potentials | comparison with scalar/invariant approaches | directional representations matter for atomic environments | implementation complexity increases |
| force training is effective | energy-force supervised experiments | gradients provide dense supervision | force labels require reliable reference calculations |
| long molecular dynamics is enabled | simulation demonstrations | learned potentials can be used beyond static prediction | stability and extrapolation remain critical |

When reading this paper, separate:

$$
\text{architecture gain}
\quad\text{from}\quad
\text{dataset size, force labels, cutoff, and training recipe}.
$$

## Implementation Reading

Useful implementation-level questions:

- Which irreducible representation orders $\ell$ are kept?
- What cutoff radius defines the atomic graph?
- How many interaction blocks are used?
- Are forces trained directly as energy gradients?
- Are neighbor lists rebuilt during dynamics?
- Are units, atom types, and reference energy conventions consistent?
- Is equivariance tested numerically after preprocessing?

For molecular and structure-based modeling, the same checklist connects to [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]] and [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]].

## Limitations

- Local cutoff graphs can miss long-range electrostatics unless the model or data pipeline handles them separately.
- Tensor features and tensor products increase implementation and memory complexity.
- Strong in-domain force accuracy does not guarantee robust extrapolation to unseen chemistries or geometries.
- Equivariance of the network does not automatically make the whole pipeline correct; neighbor construction, units, labels, and evaluation must also be checked.
- The model targets interatomic potentials, not docking, affinity prediction, or general protein-ligand ranking directly.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "Equivariance means the model knows chemistry." | Equivariance encodes geometric symmetry, not chemical coverage or label correctness. |
| "Energy accuracy alone proves a potential is useful." | Force accuracy, stability, and out-of-domain behavior matter for dynamics. |
| "Any molecular GNN is equivalent to NequIP." | NequIP's typed tensor features are the central architectural distinction. |
| "Data efficiency removes the need for careful splits." | Domain shift and extrapolation still need explicit evaluation. |

## What to Remember

NequIP is a canonical paper for the idea that molecular architecture should respect physical symmetry inside the hidden representation:

$$
\text{geometry-aware type system}
\rightarrow
\text{equivariant layers}
\rightarrow
\text{invariant energy}
\rightarrow
\text{equivariant forces}.
$$

For this wiki, keep NequIP in the architecture shelf because its lasting contribution is an architecture contract for atomistic structure modeling. Cross-link it to computational biology only when discussing molecular modeling, potentials, and simulation workflows.

## Links

- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[papers/architectures/tensor-field-networks|Tensor Field Networks]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]
- [[papers/architectures/painn|PaiNN]]
- [[papers/architectures/egnn|EGNN]]
