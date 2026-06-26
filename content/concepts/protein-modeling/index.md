---
title: Protein Modeling Concepts
tags:
  - protein-modeling
  - concepts
---

# Protein Modeling Concepts

Protein modeling concepts describe how sequences, evolutionary signals, structures, and geometric constraints become model inputs and outputs.

## Route Map

| Need | Start | Risk |
| --- | --- | --- |
| choose a protein representation | [Protein representation](/concepts/protein-modeling/protein-representation) | pooling, tokenization, MSA/template leakage |
| define protein units and groups | [Protein domain](/concepts/protein-modeling/protein-domain), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homologs crossing train/test |
| connect sequence to structure | [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment), [Residue indexing](/concepts/protein-modeling/residue-indexing) | residue mismatches, missing residues, insertion codes |
| prepare structures | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning) | chain selection, alternate locations, unresolved regions |
| model structure | [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) | benchmark leakage and coordinate-frame assumptions |
| model binding context | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) | ligand-defined pockets and apo/holo mismatch |

## Core Concepts

| Group | Notes |
| --- | --- |
| Representation | [Protein representation](/concepts/protein-modeling/protein-representation), [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment) |
| Structure | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) |
| Indexing | [Residue indexing](/concepts/protein-modeling/residue-indexing), [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment) |
| Binding | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) |

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
