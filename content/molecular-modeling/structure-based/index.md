---
title: Structure-Based Modeling
aliases:
  - computational-biology/structure-based
  - bio/structure-based-ai
  - molecular-modeling/structure-based-ai
  - research/structure-based-ai
tags:
  - computational-biology
  - structure-based-modeling
  - drug-discovery
---

# Structure-Based Modeling

Structure-based modeling은 molecular structure를 first-class object로 다룹니다. Classical docking과 conformer workflow뿐 아니라 AI-based pose generation, scoring, structure-aware generation도 포함합니다. 핵심은 모델이 molecule score만 잘 내는지가 아니라 [[entities/protein|protein]] structure, [[entities/ligand|ligand]] geometry, [[entities/protein-ligand-complex|protein-ligand complex]]를 물리적으로 그럴듯한 시스템으로 다룰 수 있는가입니다.

이 wiki에서 이 영역은 먼저 molecular modeling anchor입니다. 가장 강한 claim이 learned representation, generation, scoring, evaluation에 있을 때 AI method가 들어옵니다.

Sequence-only input이 중심이면 [[molecular-modeling/sequence-based|Sequence-based modeling]]에서 시작하고, coordinate, pocket, pose, conformer, complex가 중심이면 이 페이지에서 시작합니다.

유용한 추상화는 아래와 같습니다.

$$
\hat{y}, \hat{X}
= f_\theta(P, L, X_0)
$$

여기서 $P$는 protein 또는 pocket, $L$은 ligand, $X_0$는 initial/noisy geometry, $\hat{X}$는 predicted pose 또는 structure, $\hat{y}$는 score 또는 property입니다.

## Working Map

1. object를 정의합니다: protein, ligand, pocket, complex.
2. candidate structure를 generate 또는 refine합니다.
3. physical/chemical plausibility를 확인합니다.
4. candidate를 rank 또는 score합니다.
5. realistic split과 generalization assumption 아래에서 평가합니다.

## Structure Workflow Map

| Stage | Question | Common failure |
| --- | --- | --- |
| Receptor preparation | 어떤 structure, chain, protonation, missing residue, cofactor인가? | template 또는 ligand leakage |
| Ligand preparation | stereo, tautomer, protonation, conformer policy는 무엇인가? | inconsistent chemical state |
| Pocket definition | known, predicted, ligand-defined, blind, grid-based 중 무엇인가? | unavailable information 사용 |
| Pose generation | search, diffusion, refinement, docking, sampling 중 무엇인가? | plausible해 보이지만 strained geometry |
| Scoring | pose score, affinity, ranking, enrichment, filter 중 무엇인가? | score meaning collapse |
| Evaluation | RMSD, clash, interaction, affinity, screening, utility 중 무엇인가? | metric이 claim과 맞지 않음 |

## Core Questions

- 언제 sequence-only representation으로 충분하고, 언제 3D structure가 필요한가?
- task가 pose quality, binding affinity, molecular generation, virtual screening 중 무엇인가?
- error가 geometric, chemical, data-driven, evaluation artifact 중 어디에서 오는가?
- benchmark가 real generalization을 측정하는가, related structure memorization을 측정하는가?
- classical force field, minimization, simulation protocol이 method의 일부인가, diagnostic일 뿐인가?

## Topics

- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/ligand|Ligand]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/molecular-modeling/force-field|Force field]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]
- [[papers/sbdd/posebusters|PoseBusters]]

## Evaluation Anchors

- [[papers/sbdd/posebusters|PoseBusters]]: pose plausibility check.
- [[concepts/sbdd/pose-generation|Pose generation]]: search/sampling과 scoring 분리.
- [[concepts/evaluation/leakage|Leakage]]: data와 benchmark audit.
- [[concepts/evaluation/scaffold-split|Scaffold split]]: ligand-side generalization.
- [[concepts/evaluation/protein-family-split|Protein family split]]: protein-side generalization.

## Paper Route

Specific paper는 [[papers/sbdd/index|Structure-based modeling papers]], [[papers/protein-modeling/index|Protein modeling papers]], [[papers/generative-models/index|Generative model papers]]에 둡니다. 이 페이지는 reusable route인 object, geometry, scoring, split, evidence에 집중합니다.

## Adjacent Areas

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[molecular-modeling/docking|Docking]]
- [[molecular-modeling/geometry|Geometry]]
- [[ai/generative-models|Generative models]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[concepts/molecular-modeling/force-field|Force field]]
