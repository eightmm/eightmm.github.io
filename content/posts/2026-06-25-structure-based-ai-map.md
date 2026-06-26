---
title: 구조 기반 모델링을 어떻게 정리할 것인가
date: 2026-06-25
tags:
  - posts
  - structure-based-modeling
  - docking
---

# 구조 기반 모델링을 어떻게 정리할 것인가

이 사이트를 정리할 때 가장 먼저 잡고 싶은 축은 [[molecular-modeling/index|Computational Biology]]입니다. 특히 구조 기반 모델링에서는 모델이 단순한 sequence나 graph가 아니라 실제 3D 구조를 가진 [[entities/protein-ligand-complex|protein-ligand complex]]를 어떻게 다룰 수 있는지가 중요합니다.

현재는 세 층으로 나눠서 생각하고 있습니다.

## 1. 무엇을 다루는가

가장 먼저 구분해야 할 것은 대상입니다. [[entities/protein|Protein]], [[entities/ligand|Ligand]], [[entities/molecule|Molecule]], [[entities/protein-ligand-complex|Protein-ligand complex]]는 서로 다른 표현과 inductive bias를 요구합니다.

구조 기반 모델링에서 중요한 질문은 모델이 무엇을 보는가입니다.

- sequence만 보는가
- 2D molecular graph를 보는가
- 3D coordinates를 직접 쓰는가
- binding pocket만 자르는가
- full protein-ligand complex를 보는가
- 여러 view를 함께 쓰는가

## 2. 어떤 방법으로 다루는가

방법론은 [[ai/index|AI]]와 [[concepts/index|Concepts]]에 쌓아둘 생각입니다. 일반적인 모델 구조로는 [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/gnn|Graph neural networks]], [[concepts/architectures/mamba|Mamba]]가 있고, 구조 데이터를 다룰 때는 [[concepts/geometric-deep-learning/equivariance|Equivariance]], [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]], [[concepts/generative-models/flow-matching|Flow matching]] 같은 개념이 자주 등장합니다.

예를 들어 [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]은 적어도 세 문제로 나눠 보는 편이 좋습니다.

- candidate pose를 생성하는 문제
- pose를 ranking하거나 scoring하는 문제
- 물리적으로 말이 안 되는 구조를 걸러내는 문제

이 셋을 분리해 두면 흔한 혼동을 줄일 수 있습니다. 좋은 [[concepts/sbdd/scoring-function|scoring function]]이 있다고 해서 생성된 pose가 자동으로 화학적으로 타당해지는 것은 아닙니다.

## 3. 어떻게 평가하는가

평가는 이 wiki가 단순히 모델 이름을 모아둔 목록이 되지 않게 해주는 부분입니다. [[papers/sbdd/posebusters|PoseBusters]]는 pose plausibility를 명시적으로 보려는 출발점이라서 유용합니다. 생성된 complex가 어떤 metric 하나에서 좋아 보인다고 해서 곧바로 성공으로 보면 안 됩니다.

계속 되돌아볼 질문은 이런 것들입니다.

- ligand가 화학적으로 타당한가
- protein-ligand geometry가 그럴듯한가
- pose quality와 binding affinity를 구분했는가
- split이 실제 generalization을 테스트하는가
- 실패 원인을 geometry, chemistry, data, evaluation 중 어디로 설명할 수 있는가

이 질문들은 [[concepts/evaluation/leakage|Leakage]], [[concepts/evaluation/scaffold-split|Scaffold split]], [[concepts/evaluation/protein-family-split|Protein family split]]과도 연결됩니다.

## 블로그와 wiki를 같이 쓰는 방식

블로그 글은 한글로 씁니다. 대신 세부 지식은 영어 wiki note로 남겨서 검색, 연결, LLM 활용에 유리하게 유지합니다.

현재 첫 경로는 이렇게 잡습니다.

1. [[molecular-modeling/structure-based/index|Structure-Based Modeling]]
2. [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand Docking]]
3. [[concepts/sbdd/index|Structure-Based Drug Discovery]]
4. [[concepts/sbdd/pose-quality|Pose Quality]]
5. [[concepts/sbdd/binding-affinity|Binding Affinity]]
6. [[papers/sbdd/posebusters|PoseBusters]]

이 정도면 전체 분야를 처음부터 완벽하게 정리하려고 하기보다, 실제로 읽고 쓰면서 확장하기에 충분한 시작점입니다.

이 사이트 자체를 어떻게 운영할지는 [[posts/2026-06-25-blog-and-wiki-workflow|블로그와 위키를 같이 쓰는 방식]]에 따로 정리했습니다.
