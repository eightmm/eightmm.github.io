---
title: Pose-aware Molecular Generation
tags:
  - research
  - computational-biology
  - molecular-generation
  - structure-based-modeling
---

# Pose-aware Molecular Generation

Status: seed question. 이 페이지는 특정 모델의 성능 claim이 아니라, pose-aware generation을 검증하려면 무엇을 분리해야 하는지 정리합니다.

## Motivation

Molecular generation은 validity, novelty, property score만으로는 structure-based task의 품질을 충분히 설명하지 못합니다. Binding pocket 안에서 plausible pose를 만들 수 있는지, pose quality와 binding affinity claim을 분리할 수 있는지가 중요합니다.

## Question

Generated molecule이 단순히 valid molecule인지가 아니라, target pocket 안에서 pose-aware하게 유효한 후보인지 어떻게 평가할 수 있을까?

## Hypothesis

Structure condition을 쓰는 generative model은 ligand-only property objective보다 pose validity, interaction recovery, scaffold transfer에서 더 나은 후보를 만들 수 있습니다. 다만 이 주장은 scaffold split, protein-family split, pocket similarity leakage를 통제해야만 의미가 있습니다.

## Method Axis

| 축 | 볼 것 |
| --- | --- |
| Object | [Objects and entities](/molecular-modeling/entities), [Protein-ligand complex](/entities/protein-ligand-complex) |
| Generation | [Generative Models](/ai/generative-models), [Molecular generation](/concepts/generative-models/molecular-generation) |
| Geometry | [Geometry and symmetry](/math/geometry-symmetry), [Equivariance](/concepts/geometric-deep-learning/equivariance) |
| Evaluation | [Pose quality](/concepts/sbdd/pose-quality), [Binding affinity](/concepts/sbdd/binding-affinity), [PoseBusters](/papers/sbdd/posebusters) |

## Evidence Plan

- Public docking/pose benchmark에서 pose validity와 molecule validity를 분리합니다.
- Scaffold split과 protein-family split을 따로 보고, 가능하면 joint split을 확인합니다.
- Docking score, pose RMSD, clash, interaction recovery, synthetic accessibility를 서로 다른 claim으로 기록합니다.
- Generated sample 중 invalid, failed docking, duplicate, filtered sample의 denominator를 숨기지 않습니다.

## Project Handoff

구현으로 넘어가면 [[projects/index|Projects]]에 public-safe evaluation workflow를 둡니다. 예를 들어 generated molecule set을 받아 pose-quality report를 만드는 tool은 project artifact입니다.

## Related

- [[research/computational-biology/index|Computational Biology Research]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[ai/generative-models|Generative Models]]
- [[concepts/evaluation/leakage|Leakage]]
