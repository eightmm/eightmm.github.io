---
title: Protein Modeling Concepts
tags:
  - protein-modeling
  - concepts
---

# Protein Modeling Concepts

Protein modeling concept는 sequence, evolutionary signal, structure, geometric constraint가 model input과 output으로 바뀌는 과정을 설명합니다.

## Route Map

| Need | Start | Risk |
| --- | --- | --- |
| choose a protein representation | [Protein representation](/concepts/protein-modeling/protein-representation) | pooling, tokenization, MSA/template leakage |
| understand sequence-only pretraining | [Protein language model](/concepts/protein-modeling/protein-language-model), [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment) | likelihood confused with fitness |
| define protein units and groups | [Protein domain](/concepts/protein-modeling/protein-domain), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homologs crossing train/test |
| connect sequence to structure | [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment), [Residue indexing](/concepts/protein-modeling/residue-indexing) | residue mismatches, missing residues, insertion codes |
| prepare structures | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning) | chain selection, alternate locations, unresolved regions |
| model structure | [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) | benchmark leakage and coordinate-frame assumptions |
| model binding context | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) | ligand-defined pockets and apo/holo mismatch |

## Core Concepts

| Group | Notes |
| --- | --- |
| Representation | [Protein representation](/concepts/protein-modeling/protein-representation), [Protein language model](/concepts/protein-modeling/protein-language-model), [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment) |
| Structure | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) |
| Indexing | [Residue indexing](/concepts/protein-modeling/residue-indexing), [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment) |
| Binding | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) |

## Checks

- Is the model sequence-only, structure-only, or sequence-structure fused?
- Is sequence likelihood being treated as evidence for function, stability, or binding without downstream validation?
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
- [[concepts/protein-modeling/protein-language-model|Protein language model]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
