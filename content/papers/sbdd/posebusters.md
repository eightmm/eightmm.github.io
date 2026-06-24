---
title: PoseBusters
tags:
  - papers
  - docking
  - evaluation
---

# PoseBusters

PoseBusters is a benchmark and validation suite for checking whether generated protein-ligand poses are physically and chemically plausible. It is useful when a model appears strong by aggregate score but may produce invalid structures.

## Main Idea

Pose quality should not be judged only by a model score. Docking and generative workflows need explicit checks for molecular geometry, steric clashes, bond validity, and protein-ligand consistency.

## Why It Matters

- Helps diagnose failures in [[research/structure-based-ai/protein-ligand-docking|protein-ligand docking]].
- Separates geometric validity from [[concepts/sbdd/scoring-function|scoring function]] performance.
- Provides a public reference point for evaluating generated poses.

## Related

- [[research/structure-based-ai/index|Structure-based AI]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[research/llm-wiki|LLM Wiki]]
