---
title: PoseBusters
tags:
  - papers
  - docking
  - evaluation
status: reading
---

# PoseBusters

PoseBusters is a benchmark and validation suite for checking whether generated protein-ligand poses are physically and chemically plausible. It is useful when a model appears strong by aggregate score but may produce invalid structures.

Source:
- [Chemical Science article](https://pubs.rsc.org/en/content/articlelanding/2024/sc/d3sc04185a)
- [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/)

## Main Idea

Pose quality should not be judged only by a model score. Docking and generative workflows need explicit checks for molecular geometry, steric clashes, bond validity, and protein-ligand consistency.

PoseBusters is useful because it shifts the question from "did the model output a pose close to a reference?" to "is the generated complex physically and chemically usable enough to evaluate further?"

## Why It Matters

- Helps diagnose failures in [[research/structure-based-ai/protein-ligand-docking|protein-ligand docking]].
- Separates geometric validity from [[concepts/sbdd/scoring-function|scoring function]] performance.
- Provides a public reference point for evaluating generated poses.
- Encourages evaluation that includes both native-like binding mode and pose plausibility.

## What It Checks

The exact checklist depends on the benchmark configuration, but the important categories are:

- Ligand identity and chemical consistency.
- Bond lengths, stereochemistry, and aromatic ring planarity.
- Intramolecular strain and clashes.
- Protein-ligand clashes and unrealistic overlap.
- Whether a pose is native-like only after it also passes plausibility checks.

## Reading Questions

- Which generated poses fail because of ligand chemistry rather than docking location?
- Does a method report RMSD, validity, and success rate as separate quantities?
- Does the benchmark test generalization beyond training-like proteins or ligands?
- Are classical docking baselines included, and under what input assumptions?
- Is any post-processing or energy minimization applied before evaluation?

## Connection to This Wiki

For this site, PoseBusters is the first evaluation anchor for [[research/structure-based-ai/index|Structure-Based AI]]. It should be linked whenever a docking, pose generation, or protein-ligand structure model claims good structural performance.

## Related

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[research/llm-wiki|LLM Wiki]]
