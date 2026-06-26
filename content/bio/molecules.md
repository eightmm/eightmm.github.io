---
title: Molecules
aliases:
  - bio-ai/molecules
tags:
  - bio
  - molecules
---


# Molecules

Molecule and ligand modeling covers small-molecule identity, representation, standardization, similarity, and property prediction. The main risk is treating strings or graphs as if they were already chemically normalized objects.

$$
r_L = \phi(L, c)
$$

where $L$ is a molecular object and $c$ can include pH, assay condition, conformer source, or target context.

## Core Notes

- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]

## Structure and Chemistry

- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/substructure-search|Substructure search]]
- [[concepts/molecular-modeling/fragment-selfies|Fragment SELFIES]]

## Tasks

- [[concepts/tasks/property-prediction|Property prediction]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]

## Checks

- Has the molecule been standardized before deduplication and split construction?
- Are stereochemistry, protonation, tautomer, salts, and conformer source handled explicitly?
- Is the split scaffold-based, temporal, assay-based, or target-aware?
- Is similarity measured on the right representation for the claim?

## Related

- [[bio/index|Bio]]
- [[bio/structure-based-ai|Structure-based AI]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
