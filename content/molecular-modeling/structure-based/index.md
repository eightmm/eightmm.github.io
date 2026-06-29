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

A useful abstraction is:

$$
\hat{y}, \hat{X}
= f_\theta(P, L, X_0)
$$

where $P$ is the protein or pocket, $L$ is the ligand, $X_0$ is an initial or noisy geometry, $\hat{X}$ is a predicted pose or structure, and $\hat{y}$ is a score or property.

## Working Map

1. Define the object: protein, ligand, pocket, or complex.
2. Generate or refine a candidate structure.
3. Check physical and chemical plausibility.
4. Rank or score candidates.
5. Evaluate under realistic split and generalization assumptions.

## Structure Workflow Map

| Stage | Question | Common Failure |
| --- | --- | --- |
| Receptor preparation | which structure, chain, protonation, missing residues, cofactors? | template or ligand leakage |
| Ligand preparation | stereo, tautomer, protonation, conformer policy? | inconsistent chemical state |
| Pocket definition | known, predicted, ligand-defined, blind, or grid-based? | using unavailable information |
| Pose generation | search, diffusion, refinement, docking, or sampling? | plausible-looking but strained geometry |
| Scoring | pose score, affinity, ranking, enrichment, or filter? | score meaning is collapsed |
| Evaluation | RMSD, clash, interaction, affinity, screening, utility? | metric does not match claim |

## Core Questions

- When should a model use sequence-only representations, and when does it need 3D structure?
- Is the task about pose quality, binding affinity, molecular generation, or virtual screening?
- Which errors are geometric, chemical, data-driven, or evaluation artifacts?
- Does a benchmark measure real generalization or memorization through related structures?
- Is a classical force field, minimization, or simulation protocol part of the method or only a diagnostic?

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

- [[papers/sbdd/posebusters|PoseBusters]] for pose plausibility checks.
- [[concepts/sbdd/pose-generation|Pose generation]] for separating search/sampling from scoring.
- [[concepts/evaluation/leakage|Leakage]] for data and benchmark audit.
- [[concepts/evaluation/scaffold-split|Scaffold split]] for ligand-side generalization.
- [[concepts/evaluation/protein-family-split|Protein family split]] for protein-side generalization.

## Paper Route

Specific papers live under [[papers/sbdd/index|Structure-based modeling papers]], [[papers/protein-modeling/index|Protein modeling papers]], and [[papers/generative-models/index|Generative model papers]]. This page stays focused on the reusable route: object, geometry, scoring, split, and evidence.

## Adjacent Areas

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[molecular-modeling/docking|Docking]]
- [[molecular-modeling/geometry|Geometry]]
- [[ai/generative-models|Generative models]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[concepts/molecular-modeling/force-field|Force field]]
