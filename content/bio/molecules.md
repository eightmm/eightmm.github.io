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

## Representation Choices

| Representation | Use For | Main Risk |
| --- | --- | --- |
| SMILES / string | sequence modeling, language-model style pretraining, generation | equivalent molecules can have many strings unless canonicalization or augmentation is explicit |
| Molecular graph | property prediction, message passing, atom/bond reasoning | stereo, aromaticity, charge, and hydrogen handling can silently change the object |
| Fingerprint | similarity search, scaffold analysis, cheap baselines | fixed bits hide which chemistry caused a match |
| Conformer / 3D coordinates | docking, geometry, shape, interaction modeling | conformer source and protonation state can dominate downstream results |
| Fragment representation | scaffold editing, constrained generation, substructure search | fragment vocabulary can bias generation and evaluation |

## Standardization Contract

Before deduplication, splitting, featurization, or model comparison, specify:

$$
L_{\mathrm{raw}}
\xrightarrow{\mathrm{standardize}}
L_{\mathrm{std}}
\xrightarrow{\mathrm{featurize}}
r_L
$$

where $L_{\mathrm{raw}}$ is the input molecule record, $L_{\mathrm{std}}$ is the standardized chemical object, and $r_L$ is the model representation.

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

## Task Map

| Task | Output | Evaluation |
| --- | --- | --- |
| Property prediction | scalar, class, or calibrated probability | regression/classification metric plus scaffold or assay-aware split |
| Similarity search | ranked molecule list | retrieval metric and duplicate policy |
| Molecular generation | valid molecule samples | validity, uniqueness, novelty, diversity, constraint satisfaction |
| Docking / SBDD | pose, score, or candidate ranking | pose quality, enrichment, affinity, and leakage checks |

## Checks

- Has the molecule been standardized before deduplication and split construction?
- Are stereochemistry, protonation, tautomer, salts, and conformer source handled explicitly?
- Is the split scaffold-based, temporal, assay-based, or target-aware?
- Is similarity measured on the right representation for the claim?

## Related

- [[bio/index|Bio]]
- [[bio/structure-based-ai|Structure-based AI]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
