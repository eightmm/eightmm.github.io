---
title: Geometry and Symmetry
tags:
  - math
  - geometry
  - symmetry
---

# Geometry and Symmetry

Geometry와 symmetry는 coordinate가 이동, 회전, permutation, re-indexing될 때 무엇이 그대로 남아야 하고 무엇이 예측 가능하게 변해야 하는지 설명합니다.

$$
f(g \cdot x) = g \cdot f(x)
$$

이 식은 equivariance의 추상적인 형태입니다.

## Route Map

| Question | Start | Use for |
| --- | --- | --- |
| mathematical object는 무엇인가? | [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group) | distance, transform, group, coordinate rule |
| 무엇이 unchanged여야 하는가? | [Invariance](/concepts/geometric-deep-learning/invariance) | class, energy, affinity 같은 scalar label |
| 무엇이 predictably transform되어야 하는가? | [Equivariance](/concepts/geometric-deep-learning/equivariance) | coordinate, force, vector field, pose update |
| coordinate는 어떤 frame에 있는가? | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame), [Distance geometry](/concepts/geometric-deep-learning/distance-geometry) | structure representation과 leakage check |
| model에서는 어떻게 구현되는가? | [Geometric deep learning](/concepts/geometric-deep-learning) | equivariant GNN과 coordinate-aware architecture |

## Transformation Rules

| Object | Rule | Interpretation |
| --- | --- | --- |
| Translation | $x_i' = x_i + t$ | 많은 structure task에서 origin은 중요하지 않아야 합니다 |
| Rotation | $x_i' = R x_i$ | orientation은 scalar label을 바꾸면 안 됩니다 |
| Rigid motion | $x_i' = R x_i + t$ | coordinate가 하나의 body처럼 함께 움직입니다 |
| Permutation | $X' = P X$ | node/residue/atom order가 object를 바꾸면 안 됩니다 |
| Invariant output | $f(g\cdot x)=f(x)$ | scalar target이 그대로 유지됩니다 |
| Equivariant output | $F(g\cdot x)=\rho(g)F(x)$ | output이 input과 함께 예측 가능하게 변합니다 |

For distances:

$$
d_{ij}
=
\lVert x_i-x_j\rVert_2
=
\lVert (Rx_i+t)-(Rx_j+t)\rVert_2
$$

따라서 pairwise distance는 rigid motion에 invariant합니다.

## Invariance vs Equivariance

Transformation 아래에서 output이 바뀌면 안 되면 invariance를 씁니다.

$$
f(g\cdot x)=f(x)
$$

Output이 알려진 방식으로 변해야 하면 equivariance를 씁니다.

$$
F(g\cdot x)=\rho(g)F(x)
$$

여기서 $\rho(g)$는 output space 위 transformation의 representation입니다. Scalar label에서는 $\rho(g)$가 보통 identity입니다. Coordinate나 vector output에서는 $\rho(g)$가 rotation matrix 또는 rigid-motion action일 수 있습니다.

## Coordinate Sets

Coordinate matrix가 아래와 같을 때:

$$
X =
\begin{bmatrix}
x_1^\top \\
\cdots \\
x_N^\top
\end{bmatrix}
\in \mathbb{R}^{N\times 3}
$$

rigid motion은 아래처럼 작용합니다.

$$
X' = X R^\top + \mathbf{1}t^\top
$$

Atom 또는 residue order의 permutation은 아래처럼 작용합니다.

$$
X' = P X
$$

Updated coordinate를 예측하는 coordinate model은 보통 아래 조건을 만족해야 합니다.

$$
F(PXR^\top+\mathbf{1}t^\top)
=
P F(X) R^\top + \mathbf{1}t^\top
$$

이 식은 structure model이 indexing과 geometry를 함께 존중하는지 확인하는 compact한 방법입니다.

## Groups and Coordinates

- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/spherical-harmonics|Spherical harmonics]]

## Target Map

| Target | Expected symmetry | Examples |
| --- | --- | --- |
| Class / affinity / energy | invariant | property prediction, binding score |
| Coordinate update | equivariant | pose generation, structure refinement |
| Force / vector field | equivariant | molecular dynamics, flow matching in coordinates |
| Graph relation | permutation equivariant or invariant | contact map, interaction graph |

## AI Connections

- CNNs encode translation-related inductive bias.
- GNNs handle permutation-sensitive relational data through message passing.
- Equivariant networks are important for molecules, proteins, and coordinate prediction.
- Docking and pose generation need geometry validity, not only scalar scores.

## Model Family Map

| Model family | Built-in symmetry | Typical use |
| --- | --- | --- |
| CNN | translation equivariance on grids | images, local spatial patterns |
| GNN | permutation equivariance over nodes | molecules, residues, relational data |
| Graph Transformer | permutation-aware attention with graph bias | long-range graph interactions |
| E(3)/SE(3)-equivariant model | rigid-motion equivariance | coordinate prediction, forces, poses |
| Set model | permutation invariance/equivariance | unordered objects, pooled representations |

Model family는 dataset format만이 아니라 object와 target의 symmetry에 맞아야 합니다.

## Checks

- 어떤 transformation이 label을 보존하는가?
- 어떤 output이 input과 함께 rotate 또는 translate되어야 하는가?
- coordinate frame이 arbitrary한가, physically meaningful한가?
- relative geometry가 더 안전한데 model이 absolute coordinate를 쓰고 있지는 않은가?
- atom, residue, node permutation이 explicit하게 처리되는가?
- evaluation metric이 model target과 같은 symmetry를 존중하는가?

## Related

- [[math/index|Math]]
- [[molecular-modeling/geometry|Computational Biology geometry]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
