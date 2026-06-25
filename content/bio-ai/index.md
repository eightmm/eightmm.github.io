---
title: Bio-AI
tags:
  - bio-ai
---

# Bio-AI

Bio-AI notes focus on structure-based modeling, proteins, molecules, and genome-level sequence modeling. The scope is intentionally narrower than all of computational biology.

The recurring pattern is to model biological or chemical objects with AI functions:

$$
\hat{y} = f_\theta(x_{\mathrm{bio}}, x_{\mathrm{context}})
$$

Here $x_{\mathrm{bio}}$ may be a sequence, molecule, structure, or complex, and $x_{\mathrm{context}}$ may be a pocket, target condition, or assay context.

## Objects

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[entities/molecule|Molecule]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[entities/genome|Genome]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]

## Structure-Based AI

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[papers/sbdd/posebusters|PoseBusters]]

## Protein and Sequence Modeling

- [[research/protein-modeling/index|Protein modeling]]
- [[research/protein-modeling/mambafold|MambaFold]]
- [[research/genome-sequence-modeling/index|Genome sequence modeling]]

## Geometry and Evaluation

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]

## Related

- [[ai/index|AI]]
- [[papers/index|Papers]]
- [[projects/index|Projects]]
