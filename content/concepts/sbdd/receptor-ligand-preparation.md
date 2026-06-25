---
title: Receptor and Ligand Preparation
tags:
  - sbdd
  - docking
  - data-preprocessing
---

# Receptor and Ligand Preparation

Receptor and ligand preparation converts raw structures and molecules into model-ready inputs for docking, scoring, pose evaluation, or property prediction. Preparation choices can dominate results, so they should be treated as part of the method.

A preparation pipeline can be written as:

$$
(P_{\mathrm{raw}}, L_{\mathrm{raw}})
\xrightarrow{\phi_{\mathrm{prep}}}
(P_{\mathrm{model}}, L_{\mathrm{model}})
$$

where $\phi_{\mathrm{prep}}$ includes cleaning, standardization, protonation, conformer handling, and binding-site definition.

## Receptor Preparation

- Choose chain, biological assembly, and relevant structure.
- Handle missing residues, alternate conformations, waters, ions, metals, cofactors, and ligands.
- Assign protonation or charge states when needed.
- Define the pocket, docking box, or local context.
- Decide whether the receptor is rigid, flexible, or ensemble-based.

## Ligand Preparation

- Define [[concepts/molecular-modeling/molecular-identity|molecular identity]] before deduplication and split assignment.
- Standardize salt, charge, aromaticity, tautomer, and stereochemistry policy.
- Enumerate or choose protonation and tautomer states when relevant.
- Generate conformers if the workflow requires 3D inputs.
- Preserve identifiers without using them as model features.
- Track input hash and featurizer/preparation version.

## Checks

- Are receptor and ligand preparation versions recorded?
- Is pocket definition independent from test-set information?
- Are stereochemistry and protonation rules consistent across splits?
- Are failed preparations counted and reported?
- Are preparation failures correlated with target labels or molecular classes?

## Related

- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/systems/reproducibility|Reproducibility]]
