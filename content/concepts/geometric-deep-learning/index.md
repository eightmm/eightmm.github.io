---
title: Geometric Deep Learning
tags:
  - geometric-deep-learning
  - geometry
---

# Geometric Deep Learning

Geometric deep learning은 [[concepts/math/geometry|geometry]]와 [[concepts/math/symmetry-group|symmetry group]] 위에 세우는 AI layer입니다. Graph, coordinate, transformation, manifold 같은 구조를 존중하는 model을 다룹니다.

이 섹션은 순수 수학 교과서가 아니라 math foundation에서 protein, molecule, 3D structure, vision, graph-structured data용 neural architecture로 넘어가는 다리입니다.

핵심 질문은 model이 transformation group $G$ 아래에서 어떻게 행동해야 하는가입니다.

$$
f(g\cdot x) = \rho(g) f(x),
\qquad
g\in G
$$

여기서 $\rho(g)$는 output representation이 어떻게 변환되어야 하는지 설명합니다.

## 결정 패턴

Geometric model에서는 architecture를 고르기 전에 contract를 먼저 정합니다.

$$
(\text{object}, \text{target}, \text{group}, \text{split})
\rightarrow
\text{feature and readout design}
$$

- Object: molecule, protein, protein-ligand complex, point cloud, graph, structure.
- Target: scalar property, ranking, distance, vector field, coordinate update, pose, generated structure.
- Group: permutation, translation, rotation, reflection, SO(3), SE(3), E(3).
- Split: 필요하면 ligand scaffold, protein family, complex pair, assay/source, time.

Group 선택은 data와 deployment setting에 모두 맞아야 합니다. Preprocessing으로 강제한 symmetry는 inference time에도 같은 정보가 있을 때만 유효합니다.

## 수학 배경

| 필요 | 시작점 |
| --- | --- |
| distance, angle, coordinate basics | [Geometry](/concepts/math/geometry) |
| group actions and symmetry | [Symmetry group](/concepts/math/symmetry-group) |
| vectors, matrices, bases, eigenspaces | [Linear algebra](/concepts/math/linear-algebra) |

## Symmetry

| 질문 | 시작점 |
| --- | --- |
| should the output stay the same after a transform? | [Invariance](/concepts/geometric-deep-learning/invariance), [Invariant feature](/concepts/geometric-deep-learning/invariant-feature) |
| should the output transform with the input? | [Equivariance](/concepts/geometric-deep-learning/equivariance), [Equivariant feature](/concepts/geometric-deep-learning/equivariant-feature) |
| rotations only | [SO(3)](/concepts/geometric-deep-learning/so3) |
| rotations and translations | [SE(3)](/concepts/geometric-deep-learning/se3) |
| rotations, translations, and reflections | [E(3)](/concepts/geometric-deep-learning/e3) |

## Coordinate와 Distance

| 필요 | 시작점 | 일반적인 output |
| --- | --- | --- |
| declare coordinate source and frame | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame) | valid input contract |
| use distances without orientation | [Distance geometry](/concepts/geometric-deep-learning/distance-geometry) | invariant scalar features |
| predict or refine coordinates | [Coordinate update](/concepts/geometric-deep-learning/coordinate-update) | equivariant vector/coordinate output |

## Representations

| 필요 | 시작점 | 쓸 때 |
| --- | --- | --- |
| angular basis on directions | [Spherical harmonics](/concepts/geometric-deep-learning/spherical-harmonics) | local orientation or angular structure matters |
| scalar/vector/tensor feature types | [Irreducible representation](/concepts/geometric-deep-learning/irreducible-representation) | feature channels must transform predictably |
| equivariant message passing with typed channels | [Tensor Field Network](/concepts/geometric-deep-learning/tensor-field-network) | higher-order geometric expressivity is worth the cost |

## Model과 Operation

| 필요 | 시작점 |
| --- | --- |
| choose a geometric model family | [Geometric architecture](/concepts/geometric-deep-learning/geometric-architecture) |
| graph neural network with symmetry constraints | [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn) |

## 공개 가능한 check

- coordinate source가 experimental, predicted, docked, generated, simulated, conformer-generated 중 무엇인지 적습니다.
- chirality와 stereochemistry를 보존하는지 적습니다.
- graph construction이 deployment에서 사용할 수 있는 input만 쓰는지 적습니다.
- scalar output은 invariant인지, coordinate/vector output은 equivariant인지 적습니다.
- private structure, unpublished result, host path, internal benchmark name은 쓰지 않습니다.

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
