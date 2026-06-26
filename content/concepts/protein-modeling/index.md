---
title: Protein Modeling Concepts
tags:
  - protein-modeling
  - concepts
---

# Protein Modeling Concepts

Protein modeling concepts describe how sequences, evolutionary signals, structures, and geometric constraints become model inputs and outputs.

## Core Concepts

- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-domain|Protein domain]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]

## Checks

- Is the model sequence-only, structure-only, or sequence-structure fused?
- Is the pocket representation ligand-defined, predicted, or deployable?
- Are residue indices aligned across sequence tokens, structure residues, and coordinates?
- Are homologs and near-duplicate proteins separated across splits?
- Are missing residues, insertion codes, non-standard residues, and chain IDs handled explicitly?
- Does the evaluation measure geometry, function, interaction, or downstream task transfer?

## Related

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[entities/protein|Protein]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
