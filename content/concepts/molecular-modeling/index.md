---
title: Molecular Modeling Concepts
tags:
  - molecular-modeling
  - concepts
---

# Molecular Modeling Concepts

Molecular modeling concepts describe how small molecules become model inputs: strings, graphs, fingerprints, conformers, descriptors, and 3D coordinates.

## Core Concepts

- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]

## Data Checks

- Standardize molecules before deduplication and splitting.
- Decide whether to preserve or flatten stereochemistry.
- Record tautomer, salt, charge, protonation, and conformer protocols.
- Use scaffold or cluster splits instead of random splits for ligand-side generalization.
- Cache features with featurizer version and input hash.

## Related

- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
