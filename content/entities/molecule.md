---
title: Molecule
tags:
  - entities
  - molecule
---

# Molecule

A molecule is a collection of atoms and bonds represented as a graph, 3D conformation, or set of measured properties.

A molecule can be represented by a string, graph, fingerprint, descriptor vector, conformer ensemble, or learned embedding:

$$
r_m = \phi(m, q)
$$

where $m$ is the standardized molecular identity and $q$ is the representation protocol: SMILES tokenization, graph featurization, conformer generation, fingerprint parameters, or descriptor calculation.

## Why It Matters

- It is the base object for [[entities/ligand|ligands]], fragments, and molecular datasets.
- Molecular models must separate topology, coordinates, stereochemistry, and assay context.
- Geometry-aware methods often treat molecules as graphs with coordinate-dependent features.
- The same raw record can map to different model inputs depending on standardization, tautomer, protonation, and stereochemistry policy.

## Checks

- Which representation is used: SMILES, graph, fingerprint, conformer, or coordinates?
- Are stereochemistry, charges, and aromaticity normalized consistently?
- Is the molecule standardized before deduplication and splitting?
- Does the task require a single molecule, molecule pair, or molecule-protein context?
- Are generated molecules checked for validity, novelty, diversity, and usefulness?
- Does evaluation use scaffold or similarity groups when claiming ligand-side generalization?

## Related

- [[entities/ligand|Ligand]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[entities/assay|Assay]]
- [[entities/dataset|Dataset]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
