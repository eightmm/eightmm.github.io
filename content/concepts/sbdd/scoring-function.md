---
title: Scoring Function
tags:
  - docking
  - scoring
  - modeling
---

# Scoring Function

A scoring function assigns a value to a candidate structure, pose, or interaction. In docking, it often ranks protein-ligand poses or estimates binding-related quality.

A generic scoring function is:

$$
s = S_\theta(P, L, X)
$$

where $P$ is a protein or pocket, $L$ is a ligand, and $X$ is a candidate pose or complex geometry. Lower or higher scores may be better depending on the convention.

## Uses

- Rank candidate poses from [[research/structure-based-ai/protein-ligand-docking|protein-ligand docking]].
- Estimate or proxy [[concepts/sbdd/binding-affinity|binding affinity]].
- Filter physically implausible structures.
- Compare model outputs under a shared objective.

## Failure Modes

- Good score with bad geometry.
- Dataset bias that rewards shortcuts.
- Poor calibration outside the training distribution.
- Confusion between pose quality and binding affinity.

## Related

- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[research/structure-based-ai/index|Structure-based AI]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
