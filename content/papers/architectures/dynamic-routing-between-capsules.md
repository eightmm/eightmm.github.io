---
title: Dynamic Routing Between Capsules
aliases:
  - papers/dynamic-routing-between-capsules
  - papers/capsule-networks
tags:
  - papers
  - architectures
  - vision
  - routing
---

# Dynamic Routing Between Capsules

> A capsule architecture represents entities with vectors and uses iterative routing-by-agreement to connect part predictions to higher-level entities.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Dynamic Routing Between Capsules |
| Authors | Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton |
| Year | 2017 |
| Venue | arXiv preprint |
| arXiv | [1710.09829](https://arxiv.org/abs/1710.09829) |
| Status | verified |

## Question

Convolutional networks encode local patterns and build larger receptive fields through depth, pooling, and shared filters. The paper argues that a scalar activation does not explicitly represent the pose or instantiation parameters of an entity. It proposes capsules: groups of neurons whose vector activity represents both whether an entity exists and how it is instantiated.

The second question is how lower-level parts should send evidence to higher-level entities. Instead of using a fixed pooling or convolutional aggregation, capsules use an iterative routing-by-agreement mechanism.

The architecture changes the aggregation rule:

$$
\text{fixed spatial aggregation}
\quad\longrightarrow\quad
\text{prediction agreement and learned routing}
$$

## Main Claim

The paper introduces a multi-layer capsule system in which lower-level capsules make predictions for higher-level capsules. Higher-level capsules become active when their incoming predictions agree, and the vector orientation represents instantiation parameters while vector length represents entity presence.

The bounded claim is:

$$
\text{part predictions}
\xrightarrow{\text{routing by agreement}}
\text{entity capsule}
$$

The reported experiments show strong behavior on the tested digit-recognition tasks, including overlapping digits. They do not establish that capsules universally outperform CNNs or that vector orientation is a complete disentangled pose representation.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | feature maps or lower-level capsule vectors |
| Unit | capsule vector representing entity activity and parameters |
| Prediction | transformed lower-level capsule vote for each parent capsule |
| Routing | iterative agreement-based coupling coefficients |
| Nonlinearity | squash function controls capsule length |
| Output | class/entity capsules and optional reconstruction |
| Training | discriminative margin loss, optionally reconstruction regularization |

Let $u_i$ be a lower-level capsule. A transformation matrix produces its prediction for parent capsule $j$:

$$
\hat{u}_{j\mid i}=W_{ij}u_i
$$

The parent capsule combines votes using coupling coefficients $c_{ij}$:

$$
s_j=\sum_i c_{ij}\hat{u}_{j\mid i}
$$

and outputs a squashed vector $v_j$:

$$
v_j=\operatorname{squash}(s_j)
$$

## Capsule State

The capsule vector has two intended semantic roles:

| Quantity | Intended meaning |
| --- | --- |
| $\lVert v_j\rVert$ | probability-like presence of entity $j$ |
| direction of $v_j$ | instantiation parameters such as pose |
| components of $v_j$ | learned pose/entity coordinates, not guaranteed human labels |

The squash function is:

$$
\operatorname{squash}(s)
=
\frac{\lVert s\rVert^2}{1+\lVert s\rVert^2}
\frac{s}{\lVert s\rVert}
$$

For small $\lVert s\rVert$, the output is short; for large $\lVert s\rVert$, the length approaches one while preserving direction.

The norm should be interpreted as an activation statistic under the model's training objective, not as a calibrated probability unless separately calibrated.

## Routing by Agreement

The routing algorithm begins with logits $b_{ij}$ that represent the initial preference for sending capsule $i$ to parent $j$. Coupling coefficients are normalized over possible parents:

$$
c_{ij}
=
\operatorname{softmax}_j(b_{ij})
$$

The votes are aggregated and squashed:

$$
s_j=\sum_i c_{ij}\hat{u}_{j\mid i},
\qquad
v_j=\operatorname{squash}(s_j)
$$

Agreement between a vote and the current parent output updates the routing logits:

$$
b_{ij}
\leftarrow
b_{ij}+\hat{u}_{j\mid i}\cdot v_j
$$

After a fixed number of iterations, the final $v_j$ is passed to the next layer or class head.

The iterative loop is an algorithm inside a neural layer:

1. initialize routing logits;
2. normalize couplings over parent capsules;
3. aggregate transformed votes;
4. squash parent vectors;
5. update logits from vote-output agreement;
6. repeat for a fixed number of routing iterations.

This is different from ordinary feed-forward convolution, where aggregation weights are fixed by learned kernels and spatial connectivity.

## Transformation and Pose Sharing

If a lower-level capsule represents a part under a transformation, the matrix $W_{ij}$ maps its pose-like vector into a parent prediction. The intended inductive bias is that parts contribute consistent predictions to a whole even when their local appearance changes.

However, the transformation matrices are learned parameters, not a guarantee of exact group equivariance. A capsule vector can encode pose-related information without satisfying:

$$
f(g\cdot x)=\rho(g)f(x)
$$

for a known transformation group $g$. If exact symmetry matters, compare capsules with [[concepts/geometric-deep-learning/equivariance|equivariant architectures]] and state the group/action explicitly.

## Loss

The class capsule length can be trained with a margin loss. For class $k$ and target $T_k\in\{0,1\}$:

$$
L_k
=
T_k\max(0,m^+-\lVert v_k\rVert)^2
+
\lambda(1-T_k)\max(0,\lVert v_k\rVert-m^-)^2
$$

The total classification loss is:

$$
L_{\text{margin}}=\sum_k L_k
$$

An optional reconstruction decoder regularizes the active capsule to retain information about the input:

$$
L
=
L_{\text{margin}}
+\alpha L_{\text{reconstruction}}
$$

The reconstruction term changes the representation pressure. A comparison that removes it must state whether the observed difference comes from routing, reconstruction, or both.

## Architecture Flow

The high-level flow is:

$$
\text{image}
\rightarrow
\text{convolutional feature extraction}
\rightarrow
\text{primary capsules}
\rightarrow
\text{prediction votes}
\rightarrow
\text{routing by agreement}
\rightarrow
\text{class capsules}
$$

The first convolutional layers still provide local feature extraction. Capsules replace some later scalar feature aggregation with vector-valued entities and iterative routing.

## Evidence

| Claim | Evidence type | Boundary |
| --- | --- | --- |
| capsule vectors can represent entity activity and parameters | vector norm/orientation design and reconstruction behavior | semantic meaning of dimensions is not guaranteed |
| routing-by-agreement can connect parts to wholes | iterative vote agreement mechanism | routing may be unstable or expensive in other settings |
| overlapping digit recognition improves under the reported setup | MNIST-style experiments | narrow vision task and baseline protocol |
| dynamic routing is a useful architecture primitive | proposed layer and ablations | not evidence of universal superiority over CNNs or attention |

The paper is valuable as an architecture proposal and a failure/alternative point in the history of visual representation learning. It should not be read as a settled replacement for convolution or Transformer backbones.

## Ablation and Reading Questions

| Question | What it isolates |
| --- | --- |
| Does removing routing reduce performance? | value of agreement-based connectivity |
| How many routing iterations are used? | iterative compute versus one-pass aggregation |
| Are transformation matrices shared? | pose transfer bias versus parameter capacity |
| Is reconstruction enabled? | representation regularization versus classifier loss |
| Are comparisons parameter- and compute-matched? | routing benefit versus extra operations |
| Does performance survive larger image and object complexity? | task scaling beyond overlapping digits |

For a reproduction, log capsule dimensions, number of routing iterations, coupling entropy, transformation sharing, squash implementation, margin parameters, and reconstruction weight.

## Complexity and Systems Trade-offs

If there are $I$ lower-level capsules, $J$ parent capsules, capsule dimension $d$, and $R$ routing iterations, transforming and comparing all votes can cost roughly:

$$
O(RI J d)
$$

before accounting for spatial grouping and batching. This can be much more expensive than a single fixed convolutional aggregation.

| Property | Benefit | Cost or risk |
| --- | --- | --- |
| vector-valued entities | retains activity and pose-like information | larger feature state |
| learned routing | input-dependent part-to-whole assignment | iterative latency |
| agreement scores | encourages consistent votes | sensitive to scale and initialization |
| reconstruction | regularizes information retention | adds objective and decoder capacity |
| local capsule connectivity | controls computation | may miss long-range entity relations |

The routing loop is part of the inference cost. Treating it as a free interpretation layer understates the architecture's systems burden.

## Relation to Other Architectures

| Architecture | Relation |
| --- | --- |
| [CNN](/concepts/architectures/cnn) | scalar local features and fixed learned aggregation |
| [Pointer Networks](/papers/architectures/pointer-networks) | attention selects input positions; capsules route part votes to parent entities |
| [Set Transformer](/papers/architectures/set-transformer) | permutation-aware attention over sets without capsule pose vectors |
| [Equivariant GNNs](/concepts/geometric-deep-learning/equivariant-gnn) | explicit group transformation laws when symmetry is required |
| [Transformer](/papers/architectures/attention-is-all-you-need) | content-based token mixing with a feed-forward layer stack |

Capsules and attention both use input-dependent interactions, but their contracts differ. Routing by agreement aggregates transformed vector votes; attention computes normalized query-key scores and value mixtures.

## Limitations

- Dynamic routing adds iterative computation and can be difficult to scale.
- A vector orientation is not automatically a disentangled or equivariant pose representation.
- Coupling coefficients can be unstable, diffuse, or dominated by initialization and normalization choices.
- Results on overlapping digits do not establish broad visual or multimodal superiority.
- The margin and reconstruction losses are part of the method and complicate architecture-only comparisons.
- The model does not specify a general method for hierarchical entities of arbitrary depth.
- Routing behavior can be post-hoc interpreted too strongly; agreement is a learned score, not a proof of part-whole causality.

## Implementation Checklist

- Define capsule shape as `(batch, capsule_count, capsule_dim)` at every layer.
- Keep prediction transforms and coupling coefficients separate in logs.
- Check coupling normalization over the intended parent axis.
- Monitor routing entropy and vote-output agreement across iterations.
- Test routing with and without reconstruction under matched compute.
- State whether transformation matrices are shared spatially or by capsule type.
- Compare against a parameter- and FLOP-matched CNN baseline.
- If claiming geometric behavior, specify the transformation group and test equivariance directly.

## Why It Matters

Capsules provide a concrete alternative to the assumption that every feature should be a scalar activation:

$$
\text{entity evidence}
\rightarrow
\text{vector-valued vote}
\rightarrow
\text{agreement-based routing}
$$

Even when a project does not use capsule layers, the paper is useful for asking whether a model should represent objects, parts, poses, and relations explicitly rather than relying only on pooled scalar features.

Read it after [[papers/architectures/gradient-based-learning-applied-to-document-recognition|LeNet-5]] and [[papers/architectures/deep-residual-learning|ResNet]], then compare its routing idea with attention and geometric equivariance.

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[papers/architectures/pointer-networks|Pointer Networks]]
- [[papers/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]

---
