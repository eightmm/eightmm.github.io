---
title: Protein
tags:
  - entities
  - protein
---

# Protein

Proteins are biological macromolecules represented as sequences, structures, surfaces, graphs, or dynamics depending on the task.

A protein representation can be sequence-only, structure-only, or fused:

$$
r_P = \phi(s_{1:L}, X, c)
$$

where $s_{1:L}$ is the residue sequence, $X$ is optional structure or coordinate information, and $c$ is context such as domain, family, pocket, mutation, construct, or assay.

## Modeling Views

- Sequence view for language-model and [[concepts/learning/self-supervised-learning|self-supervised learning]] methods.
- Structure view for folding, docking, and [[concepts/geometric-deep-learning/equivariant-gnn|equivariant]] models.
- Functional view for binding sites, domains, and interactions.
- Target view for assays, docking benchmarks, and virtual screening.
- Dataset view for homolog grouping, sequence identity clustering, and protein-family splits.

## Checks

- Is the task sequence-only, structure-only, or sequence-structure combined?
- Are homologs, protein families, and near-duplicate chains separated across splits?
- Are domains, missing residues, mutations, and cofactors represented explicitly?
- Does the model predict structure, function, interaction, or a downstream assay label?
- Is the split grouped by protein family, sequence identity, target, or assay?
- Is the structure experimental, predicted, modeled, or unavailable at deployment time?
- Does the representation include information that would not be available for the intended task?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/pocket|Pocket]]
- [[entities/assay|Assay]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/protein-domain|Protein domain]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[research/protein-modeling/index|Protein modeling]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[research/protein-modeling/mambafold|MambaFold]]
