---
title: Structure-Based Modeling
aliases:
  - bio/structure-based-ai
  - bio-ai/structure-based-ai
  - molecular-modeling/structure-based-ai
  - research/structure-based-ai
tags:
  - bio
  - structure-based-ai
  - drug-discovery
---

# Structure-Based Modeling

Structure-based modeling treats molecular structure as a first-class object. It covers classical docking and conformer workflows as well as AI-based pose generation, scoring, and structure-aware generation. The central question is not just whether a model can score a molecule, but whether it can reason about [[entities/protein|protein]] structure, [[entities/ligand|ligand]] geometry, and the [[entities/protein-ligand-complex|protein-ligand complex]] as a physically plausible system.

For this wiki, this area is a molecular modeling anchor first. AI methods enter when the strongest claim is about learned representation, generation, scoring, or evaluation.

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

## Core Questions

- When should a model use sequence-only representations, and when does it need 3D structure?
- Is the task about pose quality, binding affinity, molecular generation, or virtual screening?
- Which errors are geometric, chemical, data-driven, or evaluation artifacts?
- Does a benchmark measure real generalization or memorization through related structures?

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
- [[papers/sbdd/posebusters|PoseBusters]]

## Evaluation Anchors

- [[papers/sbdd/posebusters|PoseBusters]] for pose plausibility checks.
- [[concepts/sbdd/pose-generation|Pose generation]] for separating search/sampling from scoring.
- [[concepts/evaluation/leakage|Leakage]] for data and benchmark audit.
- [[concepts/evaluation/scaffold-split|Scaffold split]] for ligand-side generalization.
- [[concepts/evaluation/protein-family-split|Protein family split]] for protein-side generalization.

## Recent Papers

- [[papers/generative-models/molexar|Molexar]] — unified multimodal molecular foundation model on Fragment-SELFIES
- [[papers/protein-modeling/multi-scale-antibody-binding|Multi-scale ML for Antibody-Antigen Binding]] — multi-scale fusion for affinity prediction, cross-pathogen transfer failure
- [[papers/protein-modeling/meet-equivariant-peptide|MEET]] — memory-efficient equivariant transformer for scalable peptide design

## Adjacent Areas

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[ai/generative-models|Generative models]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]
