---
title: SE(3)-Transformers
aliases:
  - papers/se3-transformer
  - papers/se3-transformers
  - papers/se-3-transformer
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - attention
  - equivariance
---

# SE(3)-Transformers

> The paper builds self-attention for 3D point clouds and graphs while preserving equivariance to rotations and translations.

## Metadata

| Field | Value |
| --- | --- |
| Paper | SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks |
| Authors | Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling |
| Year | 2020 |
| Venue | NeurIPS 2020 |
| arXiv | [2006.10503](https://arxiv.org/abs/2006.10503) |
| Proceedings | [NeurIPS 2020 paper](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf) |
| Official implementation | [FabianFuchsML/se3-transformer-public](https://github.com/FabianFuchsML/se3-transformer-public) |
| Status | full note started |

## One-Line Takeaway

SE(3)-Transformer adapts self-attention to 3D point clouds and graphs by making attention weights invariant and value features equivariant, so the layer respects rotations and translations by construction.

## Question

Self-attention is flexible:

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
$$

But ordinary attention does not know that a 3D structure can be rotated or translated without changing its physical identity.

For points:

$$
x_i \in \mathbb{R}^3,
$$

a global rigid transform is:

$$
x_i' = Rx_i + t,
$$

where:

$$
R \in SO(3), \qquad t \in \mathbb{R}^3.
$$

For scalar outputs such as class labels or molecular properties, the prediction should be invariant:

$$
f(RX+t)=f(X).
$$

For vector or geometric features, the output should rotate with the input:

$$
F(RX+t)=RF(X).
$$

The paper asks:

> Can self-attention operate on 3D points and graphs while guaranteeing SE(3) equivariance?

## Main Claim

SE(3)-Transformer decomposes attention into:

1. invariant attention weights;
2. equivariant value features.

If attention weights are invariant:

$$
\alpha_{ij}' = \alpha_{ij},
$$

and values are equivariant:

$$
v_j' = \rho(R)v_j,
$$

then the weighted sum is equivariant:

$$
\sum_j \alpha_{ij}' v_j'
=
\sum_j \alpha_{ij}\rho(R)v_j
=
\rho(R)\sum_j \alpha_{ij}v_j.
$$

Here $\rho(R)$ is the representation that describes how a feature type transforms under rotation.

The architecture therefore keeps the useful data-dependent weighting of attention while respecting the geometry of 3D inputs.

## Architecture Contract

| Component | Role | Symmetry Requirement |
| --- | --- | --- |
| node coordinates $x_i$ | 3D positions | transform as vectors under $SE(3)$ |
| node features | scalar/vector/tensor features | transform by representation type |
| attention logits | pairwise relevance | invariant to global pose |
| attention weights | normalized relevance | invariant |
| value features | message content | equivariant |
| output features | updated node representation | equivariant |
| pooling/readout | graph-level prediction | invariant when needed |

The layer maps a geometric graph to another geometric graph:

$$
\{x_i, f_i\}_{i=1}^{N}
\rightarrow
\{x_i, f_i'\}_{i=1}^{N},
$$

where coordinates define geometry and features transform according to their representation type.

## SE(3) Equivariance

$SE(3)$ is the group of 3D rotations and translations:

$$
x \mapsto Rx+t,
\qquad
R\in SO(3).
$$

Translation equivariance is handled by using relative coordinates:

$$
r_{ij}=x_j-x_i.
$$

Under translation:

$$
r_{ij}'=(x_j+t)-(x_i+t)=r_{ij}.
$$

Under rotation:

$$
r_{ij}'=R(x_j-x_i)=Rr_{ij}.
$$

So relative geometry is the correct object for local message passing.

## Feature Types

Equivariant models often organize features by type. A type-$l$ feature transforms according to an irreducible representation of $SO(3)$:

$$
f^{(l)} \mapsto D^{(l)}(R)f^{(l)}.
$$

Examples:

| Type | Informal Meaning | Transform Behavior |
| --- | --- | --- |
| $l=0$ | scalar | unchanged by rotation |
| $l=1$ | vector | rotates like a 3D vector |
| $l=2$ | higher-order tensor-like feature | transforms by higher-order representation |

Scalar features can represent:

- atom type;
- residue type;
- charge category;
- invariant learned hidden channels.

Vector or higher-order features can represent directional information:

- orientation;
- force-like signals;
- geometric gradients;
- local frames.

This is more expressive than scalar-only hidden features, but it is also more complex than models such as [[papers/architectures/egnn|EGNN]].

## Invariant Attention Weights

Attention weights should not change if the whole structure is rotated or translated.

For node pair $(i,j)$, the attention logit can depend on invariant quantities such as:

$$
\lVert r_{ij} \rVert,
$$

and scalar node features:

$$
f_i^{(0)}, f_j^{(0)}.
$$

Then:

$$
e_{ij}' = e_{ij}.
$$

The attention weight:

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{k\in\mathcal{N}(i)}\exp(e_{ik})}
$$

is also invariant:

$$
\alpha_{ij}'=\alpha_{ij}.
$$

The key idea is that attention decides "how much" using invariant scores.

## Equivariant Values

The values must carry geometric information in a way that transforms correctly.

A value feature for edge $(i,j)$ can be constructed using equivariant kernels:

$$
v_{ij}
=
K(r_{ij}) f_j.
$$

The kernel must satisfy an equivariance constraint:

$$
K(Rr)
=
\rho_{\text{out}}(R)K(r)\rho_{\text{in}}(R)^{-1}.
$$

Then:

$$
v_{ij}' = \rho_{\text{out}}(R)v_{ij}.
$$

The output message:

$$
f_i'
=
\sum_{j\in\mathcal{N}(i)}
\alpha_{ij}v_{ij}
$$

is equivariant because invariant weights multiply equivariant values.

## Tensor Field Network Backbone

SE(3)-Transformer builds on Tensor Field Networks. The core idea is to construct filters using spherical harmonics and radial functions:

$$
K(r)
=
\sum_{\ell}
R_\ell(\lVert r\rVert)Y_\ell(\hat{r}).
$$

where:

- $R_\ell$ is a learnable radial profile;
- $Y_\ell$ is a spherical harmonic component;
- $\hat{r}=r/\lVert r\rVert$ is direction.

This gives filters that transform correctly under rotations.

The SE(3)-Transformer contribution is to combine this equivariant filtering with self-attention-style data-dependent weighting.

## Attention Layer Sketch

For each node $i$:

1. compute invariant attention logits over neighbors;
2. normalize them with softmax;
3. compute equivariant value messages;
4. sum weighted value messages;
5. apply equivariant nonlinearities or normalization.

In compact form:

$$
f_i'
=
\sum_{j\in\mathcal{N}(i)}
\alpha_{ij}
K(r_{ij})f_j.
$$

where:

$$
\alpha_{ij}
=
\operatorname{softmax}_{j}
\left(
a(f_i,f_j,r_{ij})
\right)
$$

and $a$ is designed to be invariant.

This resembles graph attention:

$$
\text{attention weight} \times \text{message},
$$

but the message is equivariant rather than an unconstrained vector.

## Why This Is Not Ordinary Graph Attention

[[papers/architectures/graph-attention-networks|GAT]] computes learned attention over graph neighborhoods:

$$
h_i'
=
\sum_{j\in\mathcal{N}(i)}
\alpha_{ij}W h_j.
$$

This is permutation-aware, but not SE(3)-equivariant. If node features include coordinates, a vanilla GAT does not guarantee correct behavior under 3D rotation.

SE(3)-Transformer changes the message:

$$
W h_j
\rightarrow
K(r_{ij})f_j,
$$

and constrains the attention weights:

$$
\alpha_{ij}(RX+t)=\alpha_{ij}(X).
$$

The result is graph attention with a geometric contract.

## Comparison to EGNN

| Property | [[papers/architectures/egnn|EGNN]] | SE(3)-Transformer |
| --- | --- | --- |
| Symmetry group | $E(n)$ | $SE(3)$ |
| Reflection behavior | equivariant to reflections | rotation/translation equivariant |
| Hidden features | mostly scalar features plus coordinates | scalar/vector/higher-order features |
| Directional expressivity | through coordinate updates | through equivariant feature types and kernels |
| Mathematical machinery | distances and relative vectors | spherical harmonics, tensor products, representations |
| Implementation complexity | lower | higher |
| Attention | not central | core mechanism |
| Good baseline role | simple equivariant GNN | expressive equivariant attention model |

EGNN is easier to implement and often a strong first baseline. SE(3)-Transformer is a better reference when a paper claims equivariant attention, tensor features, or SE(3)-equivariant message passing.

## Comparison to Tensor Field Networks

| Property | Tensor Field Network | SE(3)-Transformer |
| --- | --- | --- |
| Equivariant kernels | yes | yes |
| Data-dependent attention | not central | central |
| Message weighting | convolution/filter style | self-attention style |
| Input type | point clouds/geometric graphs | point clouds/geometric graphs |
| Main contribution | equivariant tensor field layers | attention under SE(3) equivariance |

The paper can be read as bringing Transformer-style attention to the Tensor Field Network setting.

## Comparison to Invariant Models

Invariant models may use only distances:

$$
d_{ij}=\lVert x_i-x_j\rVert.
$$

This is enough for scalar property prediction in some settings, but it can discard directional signals needed for:

- vector fields;
- coordinate updates;
- force prediction;
- local orientation;
- geometric generation;
- interaction geometry.

SE(3)-Transformer preserves directional information through equivariant feature channels.

## Evidence Reading

The paper evaluates the model on synthetic dynamics, point cloud classification, and molecular property prediction. The key evidence is that imposing SE(3) equivariance improves robustness and performance relative to non-equivariant attention baselines and equivariant non-attention baselines in those settings.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Attention can be made SE(3)-equivariant | architecture construction and proof | symmetry can be exact by design | implementation must preserve representation rules |
| Equivariance improves robustness | rotated-input experiments | predictions behave predictably under pose changes | task must actually respect the symmetry |
| Attention improves equivariant models | comparison to equivariant non-attention baselines | data-dependent neighbor weighting helps | compute cost is higher |
| Model applies to molecules | QM9-style evaluation | 3D molecular graphs benefit from equivariant attention | benchmark is limited and conformer assumptions matter |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | 3D point cloud and graph learning |
| Input unit | nodes with 3D coordinates and features |
| Output unit | invariant labels/properties or equivariant features |
| Symmetry | $SE(3)$ equivariance |
| Core mechanism | invariant attention weights plus equivariant value embeddings |
| Mathematical tools | spherical harmonics, representation types, tensor field kernels |
| Main comparison | non-equivariant attention and equivariant non-attention models |
| Key risk | complexity, runtime, and implementation correctness |

## Molecular and Structural Modeling Reading

SE(3)-Transformer is highly relevant for molecular and protein structure modeling because it directly targets 3D geometric graphs.

Useful settings:

- molecular property prediction from 3D conformers;
- atomistic force or vector-field prediction;
- protein residue or atom graphs;
- protein-ligand interaction graphs;
- geometric refinement;
- equivariant generative models;
- architectures that need orientation-sensitive hidden features.

However, the architecture is not a full biochemical model by itself. It still needs:

- chemically meaningful graph construction;
- atom/residue features;
- bond and contact representation;
- conformer quality control;
- stereochemistry handling;
- leakage-safe splits;
- appropriate losses for scalar/vector/coordinate targets.

For this wiki, SE(3)-Transformer is the reference point for expressive equivariant attention.

## Chirality and Reflection

SE(3) covers rotations and translations, not reflections. This differs from EGNN's $E(n)$ framing.

For chiral molecules, reflection behavior matters. If a model is reflection-invariant or reflection-equivariant in the wrong way, it may fail to distinguish enantiomers.

SE(3)-equivariant architectures can preserve orientation-sensitive information more naturally than fully $E(3)$-equivariant scalar-only models, but the actual behavior depends on feature types and input representation.

## Complexity and Engineering Cost

SE(3)-Transformer is mathematically elegant but heavier than scalar GNNs.

Costs include:

- managing multiple feature types;
- spherical harmonic computation;
- tensor products or equivariant kernels;
- neighborhood graph construction;
- memory overhead for high-order channels;
- specialized normalization/nonlinearity choices;
- implementation correctness checks.

The architecture should be used when the task benefits from expressive equivariant features, not just because the input has coordinates.

## Implementation Notes

### Equivariance Tests

A minimal test should compare:

$$
F(RX+t)
$$

against:

$$
\rho(R)F(X)
$$

for equivariant outputs, or:

$$
f(RX+t)
$$

against:

$$
f(X)
$$

for invariant outputs.

The error:

$$
\lVert F(RX+t)-\rho(R)F(X)\rVert
$$

should be small up to numerical tolerance.

### Feature Type Bookkeeping

Every feature channel must have a type. Mixing scalar and vector channels as if they were ordinary hidden dimensions breaks equivariance.

### Graph Construction

The layer usually operates on a local graph:

$$
\mathcal{N}(i).
$$

Graph choices include:

- radius graph;
- k-nearest neighbors;
- molecular bond graph;
- residue contact graph;
- fully connected graph for small molecules.

Graph construction must be invariant to global rotation/translation.

### Readout

For graph-level scalar prediction, the final readout must be invariant. A common pattern:

$$
\text{equivariant layers}
\rightarrow
\text{scalar channels}
\rightarrow
\text{invariant pooling}
\rightarrow
\text{MLP}.
$$

For vector outputs, the readout must preserve equivariance.

## Failure Modes

### Broken Equivariance Through Features

If non-equivariant positional features or coordinate-frame-specific features are added, the symmetry guarantee can break.

### Incorrect Tensor Type Handling

Treating higher-order representation channels as ordinary scalar channels destroys the transformation contract.

### Overkill for Scalar Tasks

Some scalar molecular property tasks may be solved well by simpler invariant or EGNN-style models. SE(3)-Transformer may add unnecessary cost.

### Benchmark Leakage

Equivariance does not protect against split leakage, conformer leakage, scaffold leakage, or homolog leakage.

### Chirality Misinterpretation

The symmetry group and feature design must match whether mirror structures should be considered equivalent.

## Common Misreadings

### "SE(3)-Transformer is just a Transformer on 3D points."

No. The important part is not only attention; it is constrained attention where weights are invariant and values are equivariant.

### "Equivariance means the model is physically correct."

No. Equivariance encodes a symmetry. Physical correctness can also require conservation laws, force-field structure, energy consistency, and valid chemistry.

### "All 3D tasks need SE(3)-Transformer."

No. Many tasks can use simpler invariant models or EGNN-like layers. SE(3)-Transformer is most justified when directional equivariant features matter.

### "SE(3) and E(3) are interchangeable."

No. $SE(3)$ excludes reflections; $E(3)$ includes them. This distinction matters for chirality and orientation-sensitive chemistry.

## Later-Paper Checklist

When reading later equivariant attention or geometric Transformer papers, ask:

- What symmetry group is guaranteed?
- Are attention weights invariant?
- Are value features equivariant?
- Which feature types are used?
- Are spherical harmonics, tensor products, or frames involved?
- Is the graph construction symmetry-preserving?
- Is equivariance tested numerically?
- Does the task need vector/tensor features or only scalar invariance?
- How does it compare to EGNN and simpler invariant baselines?
- Are runtime and memory costs reported?
- Are molecular splits and conformer pipelines leakage-safe?

## Why It Matters

SE(3)-Transformer is a key architecture paper because it established attention as a viable operation for 3D equivariant graph learning.

Its lasting pattern is:

$$
\text{invariant attention weights}
+
\text{equivariant values}
\Rightarrow
\text{equivariant attention output}.
$$

That idea appears in many later protein, molecule, point-cloud, and geometric generative models.

## Limitations

- More complex than scalar GNNs or EGNN.
- Runtime and memory can be significant.
- Correct implementation requires representation-aware operations.
- It does not solve graph construction, conformer uncertainty, or chemical validity.
- SE(3) symmetry excludes reflections, which can be either good or bad depending on the task.
- Benchmark improvements must be separated from data, feature, and compute differences.

## Connections

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[papers/architectures/egnn|E(n) Equivariant GNN]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
