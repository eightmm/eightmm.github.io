---
title: Protein-Ligand Docking
tags:
  - docking
  - structure-based-drug-discovery
  - protein-ligand
---

# Protein-Ligand Docking

Protein-ligand docking estimates how a small molecule may bind to a protein binding site. A useful docking workflow separates pose generation, pose filtering, and affinity or ranking models.

## Working Questions

- How should pose plausibility be checked before using a [[concepts/sbdd/scoring-function|scoring function]]?
- Which failures are geometric, chemical, or data-driven?
- How should docking outputs be evaluated against physical constraints and benchmarks such as [[papers/sbdd/posebusters|PoseBusters]]?

## Pipeline Sketch

1. Prepare receptor and ligand structures.
2. Generate candidate poses.
3. Filter implausible poses using geometry and chemistry checks.
4. Rank candidates with a [[concepts/sbdd/scoring-function|scoring function]].
5. Record uncertainty and failure modes.

## Related

- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[research/protein-modeling/mambafold|MambaFold]]
- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/llm-wiki|LLM Wiki]]
