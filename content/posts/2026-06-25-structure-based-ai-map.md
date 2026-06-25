---
title: A Working Map for Structure-Based AI
date: 2026-06-25
tags:
  - posts
  - structure-based-ai
  - docking
status: draft
---

# A Working Map for Structure-Based AI

This site starts from a practical question: how should AI models reason about molecular structure when the object is not just a sequence or a graph, but a physical [[entities/protein-ligand-complex|protein-ligand complex]]?

My current map has three layers.

## 1. The Objects

The basic objects are [[entities/protein|Protein]], [[entities/ligand|Ligand]], [[entities/molecule|Molecule]], and [[entities/protein-ligand-complex|Protein-ligand complex]]. These pages stay short on purpose. They define what is being modeled and which representation choices matter.

For structure-based AI, the key representation question is whether a model sees:

- sequence only,
- 2D molecular graph,
- 3D coordinates,
- binding pocket,
- full protein-ligand complex,
- or a mixture of these views.

## 2. The Methods

The method layer lives mostly in [[concepts/index|Concepts]]. Some methods are general AI building blocks, such as [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/gnn|Graph neural networks]], and [[concepts/architectures/mamba|Mamba]]. Others are more directly tied to molecular structure, such as [[concepts/geometric-deep-learning/equivariance|Equivariance]], [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]], and [[concepts/generative-models/flow-matching|Flow matching]].

For [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]], I care about three different jobs:

- generate candidate poses,
- rank or score those poses,
- reject physically implausible structures.

Keeping those jobs separate avoids a common confusion: a good [[concepts/sbdd/scoring-function|scoring function]] does not automatically mean the generated pose is chemically valid.

## 3. The Evaluation

Evaluation is the part that prevents the wiki from becoming a list of model names. [[papers/sbdd/posebusters|PoseBusters]] is a useful starting point because it makes pose plausibility explicit. A generated complex should not be treated as successful only because it looks close by one metric.

The checks I want to keep returning to are:

- Is the ligand chemically valid?
- Is the protein-ligand geometry plausible?
- Is pose quality separated from binding affinity?
- Does the split test real generalization?
- Are failures explained as geometry, chemistry, data, or evaluation problems?

This connects to [[concepts/evaluation/leakage|Leakage]], [[concepts/evaluation/scaffold-split|Scaffold split]], and [[concepts/evaluation/protein-family-split|Protein family split]].

## How I Plan to Use This Wiki

The wiki part should stay small and linked. A paper note goes into [[papers/index|Papers]]. A reusable idea goes into [[concepts/index|Concepts]]. A domain-level question goes into [[research/index|Research]]. Blog posts like this one are just readable paths through those notes.

For now, the first path is:

1. [[research/structure-based-ai/index|Structure-Based AI]]
2. [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
3. [[papers/sbdd/posebusters|PoseBusters]]
4. [[concepts/sbdd/scoring-function|Scoring function]]

That is enough to start writing without pretending the whole field is already organized.
