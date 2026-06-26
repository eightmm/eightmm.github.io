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

- Score candidate poses produced by [[concepts/sbdd/pose-generation|pose generation]].
- Rank candidate poses from [[research/structure-based-ai/protein-ligand-docking|protein-ligand docking]].
- Estimate or proxy [[concepts/sbdd/binding-affinity|binding affinity]].
- Filter physically implausible structures.
- Compare model outputs under a shared objective.

## Failure Modes

- Good score with bad geometry.
- Dataset bias that rewards shortcuts.
- Poor calibration outside the training distribution.
- Confusion between pose quality and binding affinity.
- Score convention mismatch, where lower-is-better and higher-is-better are mixed.

## Task Boundary

A scoring function is not a pose generator:

$$
S_\theta(P,L,X)
\ne
p_\theta(X\mid P,L)
$$

The first evaluates a given pose $X$; the second samples or searches for candidate poses. A workflow may use both, but evaluation should say which component is being measured.

## Related

- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[research/structure-based-ai/index|Structure-based modeling]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
