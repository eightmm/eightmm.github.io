---
title: E(n) Equivariant Graph Neural Networks
aliases:
  - papers/egnn
  - papers/en-equivariant-gnn
  - papers/e-n-equivariant-graph-neural-networks
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - graph-neural-networks
  - equivariance
---

# E(n) Equivariant Graph Neural Networks

> The paper introduces a simple graph neural network layer that is equivariant to rotations, translations, reflections, and permutations while avoiding higher-order tensor features.

## Metadata

| Field | Value |
| --- | --- |
| Paper | E(n) Equivariant Graph Neural Networks |
| Authors | Victor Garcia Satorras, Emiel Hoogeboom, Max Welling |
| Year | 2021 |
| Venue | ICML 2021 |
| arXiv | [2102.09844](https://arxiv.org/abs/2102.09844) |
| Official implementation | [vgsatorras/egnn](https://github.com/vgsatorras/egnn) |
| Status | full note started |

## One-Line Takeaway

EGNN makes 3D-aware graph learning practical by updating coordinates through relative coordinate vectors and scalar messages, giving $E(n)$ equivariance without spherical harmonics or irreducible representations.

## Question

Many graph learning problems have both topology and coordinates:

- molecules have atoms, bonds, and 3D conformers;
- proteins have residues, contacts, and coordinates;
- physical systems have particles and positions;
- point clouds have unordered points in space.

A vanilla GNN processes node features:

$$
h_i \in \mathbb{R}^{d_h}
$$

on a graph:

$$
G=(V,E).
$$

But geometric inputs also include coordinates:

$$
x_i \in \mathbb{R}^{n}.
$$

For molecular or physical prediction, the model should not change its scalar prediction if the entire system is rotated, translated, or reflected. If coordinates transform as:

$$
x_i' = Qx_i + g,
$$

where $Q$ is an orthogonal matrix and $g$ is a translation vector, then scalar outputs should be invariant:

$$
y(X') = y(X),
$$

while coordinate outputs should be equivariant:

$$
\hat{x}_i' = Q\hat{x}_i + g.
$$

The paper asks:

> Can we build an equivariant GNN that is simple, efficient, and usable beyond 3D-specific tensor representation machinery?

## Main Claim

Use invariant distances to compute scalar messages, then use those scalar messages to scale relative coordinate vectors.

For edge $(i,j)$, compute a message:

$$
m_{ij}
=
\phi_e
\left(
h_i^l,
h_j^l,
\lVert x_i^l - x_j^l \rVert^2,
a_{ij}
\right).
$$

Then update coordinates with relative vectors:

$$
x_i^{l+1}
=
x_i^l
+
C
\sum_{j \neq i}
(x_i^l - x_j^l)
\phi_x(m_{ij}).
$$

Aggregate messages:

$$
m_i
=
\sum_{j \in \mathcal{N}(i)}
m_{ij}.
$$

Update node features:

$$
h_i^{l+1}
=
\phi_h(h_i^l, m_i).
$$

The architecture is simple because intermediate hidden features are scalar features, while coordinate equivariance comes from the coordinate update rule.

## Architecture Contract

| Object | Type | Transformation Behavior |
| --- | --- | --- |
| node feature $h_i$ | scalar feature vector | invariant to $E(n)$ transforms |
| coordinate $x_i$ | vector in $\mathbb{R}^n$ | equivariant |
| squared distance $\lVert x_i-x_j\rVert^2$ | scalar | invariant |
| relative vector $x_i-x_j$ | vector | equivariant to rotations/reflections, invariant to translations |
| edge feature $a_{ij}$ | scalar/categorical edge feature | invariant unless designed otherwise |
| output scalar | property/logit | invariant |
| output coordinate | position/velocity-like vector | equivariant |

The layer is designed for:

$$
(h^l, x^l, E)
\rightarrow
(h^{l+1}, x^{l+1}).
$$

This is different from a vanilla GNN, which usually has no coordinate update:

$$
h^{l+1} = \operatorname{GNNLayer}(h^l,E).
$$

## Symmetry Group

The paper targets $E(n)$ equivariance. The Euclidean group includes rotations, reflections, and translations:

$$
x \mapsto Qx + g,
$$

where:

$$
Q^\top Q = I.
$$

If $\det(Q)=1$, the transform is a rotation. If $\det(Q)=-1$, it includes a reflection.

For a coordinate function $F$, equivariance means:

$$
F(QX + g) = QF(X) + g.
$$

For a scalar function $f$, invariance means:

$$
f(QX + g) = f(X).
$$

In molecular modeling, this matters because the coordinate frame is arbitrary. A model should not treat the same molecule differently just because it is rotated in a PDB/SDF file.

## Why Distances Are Used

The squared distance between nodes is invariant:

$$
\lVert x_i' - x_j' \rVert^2
=
\lVert Qx_i + g - (Qx_j+g) \rVert^2
=
\lVert Q(x_i-x_j) \rVert^2.
$$

Because $Q$ is orthogonal:

$$
\lVert Qv \rVert^2 = v^\top Q^\top Q v = v^\top v = \lVert v \rVert^2.
$$

So:

$$
\lVert x_i' - x_j' \rVert^2
=
\lVert x_i - x_j \rVert^2.
$$

This lets the message MLP $\phi_e$ consume geometry without depending on coordinate frame orientation.

## Why Relative Vectors Are Used

The relative vector transforms equivariantly:

$$
x_i' - x_j'
=
(Qx_i+g) - (Qx_j+g)
=
Q(x_i-x_j).
$$

If the scalar coefficient $\phi_x(m_{ij})$ is invariant, then:

$$
(x_i' - x_j')\phi_x(m_{ij}')
=
Q(x_i-x_j)\phi_x(m_{ij}).
$$

Thus each coordinate update term rotates/reflections correctly.

Summing equivariant vectors stays equivariant:

$$
\sum_j Qv_{ij}
=
Q\sum_j v_{ij}.
$$

This is the central trick of EGNN.

## Layer Equivariance Sketch

Assume:

$$
x_i' = Qx_i + g,
$$

and node features are invariant:

$$
h_i' = h_i.
$$

The edge message uses invariant inputs:

$$
m_{ij}'
=
\phi_e
\left(
h_i',
h_j',
\lVert x_i' - x_j' \rVert^2,
a_{ij}
\right)
=
m_{ij}.
$$

Therefore coordinate coefficients are unchanged:

$$
\phi_x(m_{ij}') = \phi_x(m_{ij}).
$$

The coordinate update under transformed input is:

$$
x_i'^{\,l+1}
=
Qx_i^l + g
+
C\sum_{j\neq i}
Q(x_i^l-x_j^l)\phi_x(m_{ij}).
$$

Factor out $Q$:

$$
x_i'^{\,l+1}
=
Q
\left(
x_i^l
+
C\sum_{j\neq i}
(x_i^l-x_j^l)\phi_x(m_{ij})
\right)
+
g.
$$

So:

$$
x_i'^{\,l+1}
=
Qx_i^{l+1}+g.
$$

The coordinate update is $E(n)$ equivariant.

## Feature Update

Node features are updated from invariant messages:

$$
m_i = \sum_{j\in\mathcal{N}(i)}m_{ij}.
$$

Then:

$$
h_i^{l+1}
=
\phi_h(h_i^l,m_i).
$$

Since $h_i^l$ and $m_i$ are invariant, $h_i^{l+1}$ is invariant.

This is why EGNN usually treats node features as scalar channels and coordinates as explicit vector states.

## Edge Model

The edge message can include:

$$
m_{ij}
=
\phi_e
\left(
h_i,
h_j,
d_{ij}^2,
a_{ij}
\right),
$$

where:

$$
d_{ij}^2 = \lVert x_i-x_j\rVert^2.
$$

The optional $a_{ij}$ can encode edge attributes such as:

- bond type;
- contact type;
- graph relation;
- cutoff bucket;
- pairwise feature.

If $a_{ij}$ is not invariant, the equivariance contract can break. For molecular graphs, bond types are invariant categorical labels, so they are usually safe.

## Coordinate Model

The coordinate update is:

$$
x_i^{l+1}
=
x_i^l
+
C\sum_{j\neq i}
(x_i^l-x_j^l)\phi_x(m_{ij}).
$$

The normalization constant $C$ can control scale, often related to neighborhood size. Without scale control, coordinate updates can grow with node degree:

$$
\left\lVert
\sum_{j\in\mathcal{N}(i)}
(x_i-x_j)\phi_x(m_{ij})
\right\rVert
$$

may become large for dense graphs.

Coordinate updates make EGNN suitable for tasks where coordinates themselves must change, such as dynamics prediction or generative coordinate refinement.

## Invariant Readout

For scalar graph-level prediction, use invariant node features and a permutation-invariant readout:

$$
h_G
=
\sum_{i\in V} h_i^L
$$

or:

$$
h_G
=
\operatorname{Pool}(\{h_i^L\}_{i\in V}).
$$

Then:

$$
y = \rho(h_G).
$$

Since $h_i^L$ are invariant to coordinate transforms and pooling is permutation invariant, $y$ is invariant.

## Permutation Equivariance

EGNN also needs graph node permutation behavior. If node order is permuted by $P$, features and coordinates reorder:

$$
H' = PH,
\qquad
X' = PX.
$$

Message passing with shared functions and permutation-consistent edge indexing gives:

$$
H^{l+1'} = P H^{l+1},
\qquad
X^{l+1'} = P X^{l+1}.
$$

Graph-level pooling then gives permutation invariance.

This is why EGNN combines two symmetry ideas:

- graph permutation equivariance;
- Euclidean coordinate equivariance.

## Comparison to Vanilla GNN

| Property | Vanilla GNN | EGNN |
| --- | --- | --- |
| Node features | scalar features | scalar features plus coordinates |
| Geometry input | optional edge features | squared distances in messages |
| Coordinate update | usually none | explicit equivariant update |
| Rotation behavior | not guaranteed | guaranteed by construction |
| Translation behavior | not guaranteed unless distances used | guaranteed by relative coordinates |
| Main use | graph topology and attributes | geometric graphs |

Vanilla GNNs can learn from distances, but they do not automatically produce coordinate-equivariant outputs.

## Comparison to Tensor / Spherical Harmonic Models

Many equivariant neural networks use higher-order features:

- scalar channels;
- vector channels;
- tensor features;
- spherical harmonics;
- Clebsch-Gordan products;
- irreducible representations.

EGNN avoids this machinery. It keeps hidden features mostly scalar and updates coordinates directly.

| Property | Tensor-field style models | EGNN |
| --- | --- | --- |
| Feature types | scalars, vectors, higher-order tensors | scalar features plus coordinates |
| Mathematical machinery | representation theory | distance + relative vectors |
| Expressivity | often richer directional features | simpler, more constrained |
| Implementation | more complex | relatively simple |
| Efficiency | can be heavier | often lightweight |
| Reflection behavior | depends on representation choices | $E(n)$ includes reflections |

EGNN trades some expressive richness for simplicity and scalability.

## Comparison to SE(3)-Only Equivariance

$SE(3)$ includes rotations and translations:

$$
x \mapsto Rx + g,
\qquad
R\in SO(3).
$$

$E(3)$ also includes reflections:

$$
Q\in O(3).
$$

EGNN targets $E(n)$ rather than only $SE(3)$, so it is equivariant to reflections as well. That can be desirable or undesirable depending on chirality.

For molecules, chirality can matter. A model that is fully reflection-equivariant may not distinguish mirror-image structures unless chirality information is encoded in invariant features or the task is reflection-symmetric.

## Evidence Reading

The paper evaluates EGNN on dynamics, representation learning, and molecular property prediction. The evidence supports the claim that a simple equivariant graph layer can compete without higher-order representations.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| EGNN is $E(n)$ equivariant | architectural proof | symmetry is exact under stated assumptions | implementation features can break it |
| Simple scalar-message design works | experiments on physical and molecular tasks | higher-order tensors are not always necessary | may be less expressive for directional tasks |
| Model scales beyond 3D | formula defined for $\mathbb{R}^n$ | not tied to 3D spherical harmonics | most practical structure tasks are still 3D |
| Molecular property prediction benefits | QM9-style experiments | geometry-aware message passing is useful | dataset splits and conformer quality matter |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | geometric graph learning |
| Input unit | graph nodes with coordinates |
| Output unit | invariant scalar, node features, or equivariant coordinates |
| Symmetry | $E(n)$ equivariance and graph permutation equivariance |
| Core mechanism | invariant distance messages plus equivariant coordinate updates |
| Main comparison | standard GNNs and equivariant geometric models |
| Key hyperparameters | graph construction, hidden width, layers, coordinate update scale |
| Not the claim | complete physical simulation or guaranteed chirality handling |

## Molecular and Structural Modeling Reading

EGNN is directly relevant to molecular and protein modeling because it sits at the intersection of graph topology and 3D coordinates.

Useful settings:

- molecular property prediction from conformers;
- ligand graph encoding with coordinates;
- protein residue graph encoding;
- pocket-ligand interaction graphs;
- particle dynamics;
- coordinate refinement modules;
- equivariant generative models.

Important caveats:

- molecular graphs require chemically valid atom/bond features;
- protein structures need residue/atom identity, chain context, and missing-atom handling;
- graph construction can dominate model behavior;
- chirality and stereochemistry need explicit care;
- conformer uncertainty can change labels and geometry;
- equivariance does not prevent data leakage.

For structure-based AI, EGNN is often a reasonable first equivariant baseline before moving to heavier tensor-field or SE(3)-Transformer-style architectures.

## Graph Construction

The architecture assumes a graph $E$. In molecular settings, this graph can be:

- covalent bond graph;
- radius graph;
- k-nearest-neighbor graph;
- residue contact graph;
- bipartite protein-ligand graph;
- fully connected graph for small systems.

The choice affects:

$$
\mathcal{N}(i),
$$

and therefore which interactions can be represented in one layer.

If the graph is too sparse, long-range geometric effects may require many layers. If it is too dense, compute and coordinate update scale can become problematic.

## Coordinate Update Risks

Coordinate updates are powerful but risky.

### Scale Explosion

If many neighbors contribute:

$$
\Delta x_i
=
C\sum_j (x_i-x_j)\phi_x(m_{ij})
$$

can become large. Normalization and careful initialization matter.

### Coordinate Drift

Repeated coordinate updates may move points away from physically meaningful configurations unless constrained by the task or loss.

### Chirality

Because EGNN is $E(n)$ equivariant, reflection symmetry is built in. If the target distinguishes enantiomers, invariant scalar features may need stereochemical information.

### Edge Leakage

If graph edges are built using label-derived or future information, equivariance does not protect the evaluation.

## Implementation Notes

### Tensor Shapes

For batch-free notation:

$$
H \in \mathbb{R}^{N \times d_h},
\qquad
X \in \mathbb{R}^{N \times n}.
$$

For edge list:

$$
E = \{(s_k,t_k)\}_{k=1}^{M}.
$$

A typical implementation gathers:

$$
h_s, h_t, x_s, x_t.
$$

Distances:

$$
d_{st}^2 = \lVert x_s-x_t\rVert^2.
$$

Messages:

$$
m_{st} = \phi_e(h_s,h_t,d_{st}^2,a_{st}).
$$

Scatter-sum to targets:

$$
m_t = \sum_{s:(s,t)\in E}m_{st}.
$$

Coordinate deltas:

$$
\Delta x_t = \sum_{s:(s,t)\in E}(x_t-x_s)\phi_x(m_{st}).
$$

### Masks and Padding

For batched molecules/proteins, masks must prevent cross-example edges. A single accidental edge across molecules breaks the physical meaning of the graph.

### Coordinate Units

Coordinate scaling matters. Angstrom-scale molecular coordinates and normalized synthetic coordinates can lead to different distance distributions. The distance MLP sees:

$$
\lVert x_i-x_j\rVert^2.
$$

If units change, the input distribution changes.

### Edge Direction

Even for undirected graphs, implementations often store both directions:

$$
(i,j), (j,i).
$$

The message functions are shared, but the relative vectors switch sign. This is usually what is needed for coordinate updates.

## Common Misreadings

### "EGNN is just a GNN with distances."

No. Distances make messages invariant, but the coordinate update with relative vectors is what gives coordinate equivariance.

### "Equivariance means the model understands chemistry."

No. Equivariance only encodes a symmetry. Chemical validity still needs atom types, bonds, stereochemistry, charges, conformer handling, and task-appropriate labels.

### "Reflection equivariance is always good for molecules."

Not always. Chirality-sensitive tasks can require careful feature design or a symmetry group that does not erase mirror distinctions.

### "No higher-order representations means no directionality."

EGNN uses relative coordinate vectors for coordinate updates, so direction affects coordinate evolution. But hidden feature channels are scalar, which limits some directional expressivity compared with vector/tensor feature models.

## Later-Paper Checklist

When reading later geometric GNN or structure-modeling papers, ask:

- Which symmetry group is guaranteed: $E(3)$, $SE(3)$, translation only, or none?
- Are outputs invariant or equivariant?
- Are hidden features scalar, vector, tensor, or mixed?
- Does the model update coordinates or only features?
- How is the graph constructed?
- Are distances, directions, angles, or frames used?
- Does the method handle chirality?
- Are evaluations split to avoid structure or sequence leakage?
- Is equivariance exact by construction or approximate by augmentation?
- Is the architecture compared to EGNN as a simple baseline?

## Why It Matters

EGNN is a key architecture paper because it gives a simple baseline for geometric graph learning:

$$
\text{invariant scalar messages}
+
\text{equivariant coordinate updates}.
$$

It is especially valuable for this wiki because it connects:

- [[concepts/architectures/gnn|Graph neural networks]];
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]];
- molecular property prediction;
- protein and ligand coordinate modeling;
- equivariant generative modeling.

For many structure-based AI papers, EGNN is the right reference point before more complex equivariant architectures.

## Limitations

- Scalar hidden features may limit directional expressivity.
- Reflection equivariance can be problematic for chirality-sensitive tasks.
- Graph construction strongly affects performance.
- Coordinate updates can drift or become unstable.
- The architecture does not enforce chemical validity.
- Molecular benchmarks can be sensitive to conformer generation, split strategy, and label definition.

## Connections

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[papers/architectures/gcn|GCN]]
- [[papers/architectures/graph-attention-networks|Graph Attention Networks]]
- [[papers/architectures/neural-ode|Neural ODE]]
- [[papers/architectures/index|Architecture papers]]
