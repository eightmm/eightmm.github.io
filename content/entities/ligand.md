---
title: Ligand
tags:
  - entities
  - ligand
  - small-molecule
---

# Ligand

A ligand is a small molecule or molecular fragment considered as a binding partner for a protein target.

## Modeling Views

- Graph view for atoms, bonds, and molecular topology.
- Conformer view for 3D geometry and pose generation.
- Interaction view for [[research/structure-based-ai/protein-ligand-docking|protein-ligand docking]].

## Checks

- Is stereochemistry represented and preserved?
- Is the model using 2D topology, 3D coordinates, conformers, or all of them?
- Are protonation, tautomer, and charge assumptions explicit?
- Is the ligand evaluated alone or inside a [[entities/protein-ligand-complex|protein-ligand complex]]?

## Related

- [[entities/pocket|Pocket]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[papers/sbdd/posebusters|PoseBusters]]
