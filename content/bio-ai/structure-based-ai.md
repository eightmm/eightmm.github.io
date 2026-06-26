---
title: Structure-Based AI
tags:
  - bio-ai
  - structure-based-ai
---

# Structure-Based AI

Structure-based AI uses protein, ligand, pocket, and complex geometry as model input or output. This is the main Bio-AI area for docking, pose generation, scoring, and structure-aware generation.

$$
\hat{y}, \hat{X}
=
f_\theta(P, L, X_0, c)
$$

where $P$ is a protein or pocket, $L$ is a ligand, $X_0$ is optional initial geometry, and $c$ is task context.

## Core Path

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]

## Interaction and Scoring

- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]

## Evaluation Risks

- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]

## Checks

- Is the model generating a pose, ranking poses, predicting affinity, or screening ligands?
- Is the pocket known, predicted, ligand-defined, or searched blindly?
- Are geometry validity and task utility evaluated separately?
- Does evaluation require new scaffold, new protein family, or new complex generalization?

## Related

- [[bio-ai/molecules|Molecules]]
- [[bio-ai/proteins|Proteins]]
- [[bio-ai/geometry|Geometry]]
