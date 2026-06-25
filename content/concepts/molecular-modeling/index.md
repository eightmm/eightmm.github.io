---
title: Molecular Modeling Concepts
tags:
  - molecular-modeling
  - concepts
---

# Molecular Modeling Concepts

Molecular modeling concepts describe how small molecules become model inputs: strings, graphs, fingerprints, conformers, descriptors, and 3D coordinates.

## Core Concepts

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[concepts/molecular-modeling/substructure-search|Substructure search]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]

## Data Checks

- Standardize molecules before deduplication and splitting.
- Decide whether to preserve or flatten stereochemistry.
- Record tautomer, salt, charge, protonation, and conformer protocols.
- Use scaffold or cluster splits instead of random splits for ligand-side generalization.
- Cache features with featurizer version and input hash.
- Use one molecular featurization contract across train, evaluation, and inference.

## Related

- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
