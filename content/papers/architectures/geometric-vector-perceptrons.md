---
title: Learning from Protein Structure with Geometric Vector Perceptrons
aliases:
  - papers/geometric-vector-perceptrons
  - papers/gvp
  - papers/learning-from-protein-structure-with-gvp
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - protein-modeling
---

# Learning from Protein Structure with Geometric Vector Perceptrons

> The paper introduced Geometric Vector Perceptrons, a neural block that processes scalar and vector channels together so protein structure GNNs can reason over both relational graph structure and 3D geometry.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Learning from Protein Structure with Geometric Vector Perceptrons |
| Authors | Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael J. L. Townshend, Ron O. Dror |
| Year | 2021 |
| Venue | ICLR 2021 |
| arXiv | [2009.01411](https://arxiv.org/abs/2009.01411) |
| OpenReview | [1YLJDvSx6J4](https://openreview.net/forum?id=1YLJDvSx6J4) |
| Code | [drorlab/gvp](https://github.com/drorlab/gvp) |
| Status | full note started |

## One-Line Takeaway

GVP is a compact scalar-vector layer for 3D biomolecular graphs: scalar channels stay invariant, vector channels rotate with the structure, and both can be mixed inside a GNN without the full tensor-product machinery of heavier equivariant models.

## Question

Protein structure learning needs two kinds of information:

| Information | Example | Representation pressure |
| --- | --- | --- |
| relational structure | residue graph, contact graph, sequence adjacency | graph message passing |
| geometric structure | backbone directions, edge orientations, local frames | rotation-aware features |

The architecture question is:

$$
\text{Can a graph neural network carry directional 3D information without voxelizing the protein or using high-order tensor fields?}
$$

GVP answers by replacing ordinary scalar-only MLP blocks with scalar-vector perceptrons:

$$
(s,V)
\mapsto
(s',V'),
$$

where $s$ contains invariant scalar features and $V$ contains vector features that transform with the 3D coordinate frame.

## Main Claim

GVP-equipped GNNs can jointly model protein graph topology and Euclidean geometry, improving protein model quality assessment and computational protein design over graph-only or voxel-style alternatives in the paper's experiments.

The durable architecture pattern is:

$$
\text{protein structure graph}
\rightarrow
\text{scalar/vector node and edge features}
\rightarrow
\text{GVP message passing}
\rightarrow
\text{structure-aware prediction}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | protein backbone structure represented as residues with 3D coordinates |
| Unit | residue-level graph nodes and geometric edges |
| Hidden state | scalar channels $s$ plus vector channels $V$ |
| Main block | Geometric Vector Perceptron used inside graph message passing |
| Symmetry | scalar outputs invariant; vector outputs equivariant to rotations/reflections |
| Tasks in paper | model quality assessment and computational protein design |
| Domain | macromolecular structure learning |

The central contract is:

$$
\mathrm{GVP}(s,RV)
=
(s',RV')
$$

for an orthogonal transformation $R$. Scalars are unchanged; vectors rotate or reflect consistently with the input structure.

## Scalar and Vector Channels

An ordinary dense layer usually maps only scalar features:

$$
h'
=
\phi(Wh+b).
$$

For protein geometry, some features are not scalars. A backbone direction, edge direction, normal vector, or local orientation cue should transform as:

$$
v'
=
Rv.
$$

GVP separates these two roles:

$$
s \in \mathbb{R}^{n_s},
\qquad
V \in \mathbb{R}^{n_v\times 3}.
$$

Here $n_s$ is the number of scalar channels and $n_v$ is the number of vector channels. Each vector channel is a 3D vector.

The output has the same form:

$$
s' \in \mathbb{R}^{m_s},
\qquad
V' \in \mathbb{R}^{m_v\times 3}.
$$

This makes the layer a drop-in replacement for scalar MLP components when the surrounding graph layer wants to carry both invariant and equivariant features.

## Core Operation

A simplified GVP can be read as three operations.

First, linearly mix vector channels while preserving the 3D coordinate axis:

$$
V_h
=
W_h V,
\qquad
V_\mu
=
W_\mu V_h.
$$

The weights mix channels, not spatial coordinates. This is why rotation behavior is preserved:

$$
W_h(RV)
=
R(W_hV).
$$

Second, turn vector norms into scalar information:

$$
\nu
=
\lVert V_h\rVert_2.
$$

Vector norms are invariant under rotations:

$$
\lVert RV_h\rVert_2
=
\lVert V_h\rVert_2.
$$

Third, update scalar channels and gate vector channels:

$$
s'
=
\phi(W_m[s,\nu]+b),
$$

$$
V'
=
\sigma^+(\cdot)\odot V_\mu.
$$

The exact implementation has additional choices for nonlinearities and gating, but the key idea is simple:

$$
\text{vectors provide invariant norms to scalars,}
\qquad
\text{scalars control vector magnitudes.}
$$

Direction is preserved through vector channels; magnitude and feature selection are learned through scalar gates.

## Why Equivariance Holds

Let $R$ be a rotation or reflection in 3D. Vector channels transform as:

$$
V
\mapsto
RV.
$$

Channel mixing commutes with $R$ because the learned weights act over feature channels:

$$
W(RV)
=
R(WV).
$$

Norms are invariant:

$$
\lVert Rv\rVert_2
=
\lVert v\rVert_2.
$$

Scalar nonlinearities consume only scalar inputs and vector norms, so scalar outputs are invariant:

$$
s'(s,RV)
=
s'(s,V).
$$

Vector outputs are scaled by invariant gates:

$$
V'(s,RV)
=
R V'(s,V).
$$

So the layer satisfies:

$$
\mathrm{GVP}(s,RV)
=
(s',RV').
$$

This is weaker and cheaper than full tensor-field representation theory, but it captures a useful part of 3D geometry for protein structure graphs.

## GVP Message Passing

Inside a protein graph, each node has scalar/vector features:

$$
(s_i,V_i).
$$

Each edge also has geometric information, such as distance, direction, relative sequence position, or local orientation features:

$$
e_{ij}
=
(s_{ij},V_{ij}).
$$

A GVP-GNN message can be read as:

$$
m_{ij}
=
\mathrm{GVP}_m(s_i,V_i,s_j,V_j,s_{ij},V_{ij}),
$$

then messages are aggregated:

$$
\bar{m}_i
=
\sum_{j\in\mathcal{N}(i)} m_{ij},
$$

and the node state is updated:

$$
(s_i',V_i')
=
\mathrm{GVP}_u(s_i,V_i,\bar{m}_i).
$$

The aggregation is permutation-invariant over neighbors, while the vector part remains equivariant under global coordinate transformations.

## Relation to Local Frames

Protein structure models often convert geometric cues into invariant numbers by defining local coordinate frames and then measuring angles or directions inside those frames.

That can work, but it changes the representation burden:

| Route | What it stores | Risk |
| --- | --- | --- |
| invariant distances/angles only | scalar geometry | loses some directional information |
| local frame features | scalarized local directions | frame construction can be brittle |
| full tensor features | typed irreducible representations | mathematically expressive but heavier |
| GVP scalar/vector features | invariant scalars plus equivariant vectors | simpler, but limited to vector channels |

GVP sits in the practical middle: more geometric than scalar-only GNNs, lighter than high-order equivariant networks.

## Relation to Other Architecture Notes

| Paper | Similarity | Difference |
| --- | --- | --- |
| [Tensor Field Networks](/papers/architectures/tensor-field-networks) | typed equivariant channels | TFN supports higher-order tensor features with spherical harmonics |
| [SE(3)-Transformer](/papers/architectures/se3-transformer) | 3D equivariant geometric graph learning | SE(3)-Transformer adds attention and tensor-field machinery |
| [EGNN](/papers/architectures/egnn) | simple geometric GNN with equivariance | EGNN updates coordinates directly; GVP carries vector features |
| [PaiNN](/papers/architectures/painn) | scalar/vector molecular features | PaiNN is atomistic molecular message passing; GVP is framed for protein structure graphs |
| [DimeNet](/papers/architectures/dimenet) | directional geometry for molecules | DimeNet uses angular basis functions rather than vector feature channels |
| [SchNet](/papers/architectures/schnet) | structure-aware molecular graph modeling | SchNet is distance-filtered and scalar-focused |

## Evidence to Read

| Evidence | What it supports | What it does not prove |
| --- | --- | --- |
| model quality assessment results | scalar-vector structure graphs can improve protein quality scoring | universal superiority over all later equivariant models |
| computational protein design results | GVP can support sequence prediction from backbone structure | complete protein design pipeline quality |
| comparisons to graph/voxel baselines | graph plus vector geometry is useful | fairness across all representation choices and training budgets |
| equivariance property | block has the intended scalar/vector transformation behavior | all implementation features preserve symmetry automatically |

## Why This Matters

For protein and structure-based modeling, GVP is a useful reference architecture because it answers a recurring practical question:

$$
\text{How much geometric equivariance do we need before the model becomes too heavy?}
$$

GVP's answer is:

$$
\text{keep vectors as vectors,}
\quad
\text{use norms for scalar gates,}
\quad
\text{run this inside a graph network.}
$$

This pattern appears in later protein structure encoders, inverse-folding models, and geometry-aware protein representation systems.

## Limitations

GVP is not a full general-purpose equivariant tensor algebra. It mainly handles scalar and vector channels. Tasks requiring rich orientation-sensitive higher-order quantities may benefit from tensor-field or SE(3)-Transformer-style models.

The benchmark scope is also domain-specific. The paper's evidence is strongest for protein model quality assessment and computational protein design under its experimental setup. It should not be read as proving that GVP is always better than invariant geometry features or heavier equivariant networks.

The layer preserves symmetry only if the input features and implementation respect the stated scalar/vector contract. Adding frame-dependent scalar features, arbitrary coordinate components, or non-equivariant preprocessing can break the guarantee.

## Common Misreadings

### "GVP is just a GNN with coordinates."

No. The important part is that coordinates are not treated as ordinary scalar features. Directional quantities are carried as vector channels with a defined transformation rule.

### "GVP replaces SE(3)-Transformer."

No. GVP is a lighter scalar/vector architecture. SE(3)-Transformer is more expressive but heavier because it uses tensor-field machinery and equivariant attention.

### "Equivariance means the output is always rotation-invariant."

No. Scalars are invariant; vectors are equivariant. If a protein rotates, vector outputs should rotate with it.

## What to Remember

When reading protein-structure architecture papers, ask:

- Does the model distinguish scalar and vector features?
- Are vector channels transformed consistently under rotation/reflection?
- Are geometric features scalarized too early?
- Does the message passing preserve permutation behavior over graph neighbors?
- Is the model using vector features, coordinate updates, tensor products, or local frames?
- Are the benchmarks testing the architecture, the representation, or the task pipeline?

The compact mental model is:

$$
\text{GVP}
=
\text{MLP-like scalar/vector block}
+
\text{equivariant vector gating}
+
\text{protein graph message passing}.
$$

## Links

- [[papers/architectures/tensor-field-networks|Tensor Field Networks]]
- [[papers/architectures/se3-transformer|SE(3)-Transformer]]
- [[papers/architectures/egnn|EGNN]]
- [[papers/architectures/painn|PaiNN]]
- [[papers/architectures/dimenet|DimeNet]]
- [[papers/architectures/schnet|SchNet]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
