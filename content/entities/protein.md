---
title: Protein
tags:
  - entities
  - protein
---

# Protein

Proteins are biological macromolecules represented as sequences, structures, surfaces, graphs, or dynamics depending on the task.

## Modeling Views

- Sequence view for language-model and [[concepts/learning/self-supervised-learning|self-supervised learning]] methods.
- Structure view for folding, docking, and [[concepts/geometric-deep-learning/equivariant-gnn|equivariant]] models.
- Functional view for binding sites, domains, and interactions.
- Target view for assays, docking benchmarks, and virtual screening.

## Checks

- Is the task sequence-only, structure-only, or sequence-structure combined?
- Are homologs, protein families, and near-duplicate chains separated across splits?
- Are domains, missing residues, mutations, and cofactors represented explicitly?
- Does the model predict structure, function, interaction, or a downstream assay label?
- Is the split grouped by protein family, sequence identity, target, or assay?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/pocket|Pocket]]
- [[entities/assay|Assay]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[research/protein-modeling/index|Protein modeling]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[research/protein-modeling/mambafold|MambaFold]]
