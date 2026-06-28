---
title: Molecules
aliases:
  - computational-biology/molecules
  - bio/molecules
tags:
  - computational-biology
  - molecules
---


# Molecules

Molecule and ligand modeling covers small-molecule identity, representation, standardization, similarity, and property prediction. The main risk is treating strings or graphs as if they were already chemically normalized objects.

$$
r_L = \phi(L, c)
$$

where $L$ is a molecular object and $c$ can include pH, assay condition, conformer source, or target context.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| What is the modeled chemical object? | [Molecule](/entities/molecule), [Ligand](/entities/ligand), [Molecular identity](/concepts/molecular-modeling/molecular-identity) | salts, mixtures, stereo, protonation, tautomer policy |
| How is the molecule normalized? | [Molecular standardization](/concepts/molecular-modeling/molecular-standardization), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) | deduplication before standardization |
| What does the model actually see? | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract), [RDKit](/concepts/molecular-modeling/rdkit), [SMILES](/concepts/molecular-modeling/smiles), [Molecular graph](/concepts/molecular-modeling/molecular-graph) | string augmentation or graph construction changing the object |
| Is similarity meaningful for the claim? | [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint), [Molecular similarity](/concepts/molecular-modeling/molecular-similarity) | using a cheap proxy as if it were chemical equivalence |
| Is generation constrained by chemistry? | [Fragment-SELFIES](/concepts/molecular-modeling/fragment-selfies), [Molecular generation](/concepts/generative-models/molecular-generation) | validity without utility or novelty boundary |

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

## RDKit Protocol Layer

RDKit is often the code layer behind this contract. The important point is not "use RDKit" but "record the exact molecular protocol."

$$
r_L
=
F_{\psi}(\operatorname{RDKitProtocol}(L_{\mathrm{raw}}))
$$

where $\psi$ includes featurizer settings such as fingerprint radius, bit length, chirality flag, descriptor list, conformer seed, and failure policy.

| RDKit Output | Typical Use | Must Record |
| --- | --- | --- |
| canonical SMILES | deduplication, exact identity, sequence model input | standardization and stereo policy |
| Morgan fingerprint | baseline, similarity, scaffold analysis | radius, bit length, count/binary, chirality |
| descriptor vector | small-data baseline, interpretable model | descriptor list, NaN policy, scaling |
| molecular graph | GNN input | atom/bond features, hydrogens, aromaticity, charge, stereo |
| conformer | 3D model, docking, shape | protonation, seed, force field, minimization, failure rate |

## Structure and Chemistry Routes

| Area | Start | Use For |
| --- | --- | --- |
| Conformer state | [Conformer](/concepts/molecular-modeling/conformer), [Stereochemistry](/concepts/molecular-modeling/stereochemistry), [Protonation state](/concepts/molecular-modeling/protonation-state), [Tautomer](/concepts/molecular-modeling/tautomer) | making the chemical state explicit before 3D or assay claims |
| Substructure and fragments | [Substructure search](/concepts/molecular-modeling/substructure-search), [Fragment-SELFIES](/concepts/molecular-modeling/fragment-selfies) | scaffold editing, filtering, constrained generation |
| Physics-inspired preparation | [Force field](/concepts/molecular-modeling/force-field), [Energy minimization](/concepts/molecular-modeling/energy-minimization), [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) | coordinate refinement, conformer protocols, simulation-derived features |
| Prediction and retrieval | [Property prediction](/concepts/tasks/property-prediction), [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction), [Similarity search](/concepts/tasks/similarity-search) | supervised endpoints and ranked candidate lists |

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
- Are force-field, minimization, or simulation protocols recorded when coordinates are modified?
- Is the split scaffold-based, temporal, assay-based, or target-aware?
- Is similarity measured on the right representation for the claim?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[concepts/molecular-modeling/rdkit|RDKit]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
