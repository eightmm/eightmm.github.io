---
title: Geometric Inductive Bias
tags:
  - research
  - architectures
  - geometric-deep-learning
---

# Geometric Inductive Bias

Status: seed question. 이 페이지는 결과 claim이 아니라, 공개 논문과 공개 benchmark로 검증할 수 있는 비교 질문을 정리합니다.

## Motivation

Protein, molecule, structure data는 단순 sequence나 unordered token list가 아닙니다. Translation, rotation, permutation 같은 symmetry를 어떻게 처리하는지가 model efficiency와 generalization에 영향을 줍니다.

## Question

Geometry-aware architecture는 어떤 조건에서 generic Transformer보다 더 나은 inductive bias를 제공하는가?

## Hypothesis

Coordinate나 graph relation이 task claim의 핵심일 때는 equivariant 또는 invariant structure를 명시한 model이 data efficiency와 OOD generalization에서 유리할 수 있습니다. 반대로 충분한 data와 큰 model, strong pretraining이 있으면 explicit geometry bias의 이점이 줄어들 수 있습니다.

## Method Axis

| 축 | 볼 것 |
| --- | --- |
| Architecture | [Architectures](/ai/architectures), [Transformer](/concepts/architectures/transformer), [Graph neural networks](/concepts/architectures/gnn) |
| Geometry | [Geometry and symmetry](/math/geometry-symmetry), [Equivariance](/concepts/geometric-deep-learning/equivariance) |
| Object | [Objects and entities](/molecular-modeling/entities), [Structure](/entities/structure) |
| Evaluation | [Evaluation](/ai/evaluation), [OOD generalization](/concepts/evaluation/ood-generalization) |

## Evidence Plan

- 같은 input object에서 generic Transformer, GNN, equivariant model을 비교하는 public paper를 모읍니다.
- Metric만 보지 않고 data size, compute, augmentation, pretraining 여부를 같이 기록합니다.
- Claim을 IID performance, scaffold/protein-family transfer, coordinate accuracy, physical validity로 분리합니다.
- Architecture advantage가 task-specific인지, representation choice 때문인지, data leakage 때문인지 구분합니다.

## Project Handoff

실제 benchmark runner나 비교 table을 만들게 되면 [[projects/index|Projects]]로 넘깁니다. Research note는 어떤 비교가 의미 있는지 정하는 문제 정의에 머뭅니다.

## Related

- [[research/architectures/index|Architecture Research]]
- [[ai/architectures|Architectures]]
- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
