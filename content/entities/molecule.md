---
title: Molecule
tags:
  - entities
  - molecule
---

# Molecule

A molecule is a collection of atoms and bonds represented as a graph, 3D conformation, or set of measured properties.

## Why It Matters

- It is the base object for [[entities/ligand|ligands]], fragments, and molecular datasets.
- Molecular models must separate topology, coordinates, stereochemistry, and assay context.
- Geometry-aware methods often treat molecules as graphs with coordinate-dependent features.

## Checks

- Which representation is used: SMILES, graph, fingerprint, conformer, or coordinates?
- Are stereochemistry, charges, and aromaticity normalized consistently?
- Does the task require a single molecule, molecule pair, or molecule-protein context?
- Are generated molecules checked for validity, novelty, diversity, and usefulness?

## Related

- [[entities/ligand|Ligand]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[entities/assay|Assay]]
- [[entities/dataset|Dataset]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
