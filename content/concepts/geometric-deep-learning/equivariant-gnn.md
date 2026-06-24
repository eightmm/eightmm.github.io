---
title: Equivariant GNN
tags:
  - geometric-deep-learning
  - gnn
  - equivariance
---

# Equivariant GNN

An equivariant graph neural network preserves known geometric transformations in its outputs. For molecular modeling, this usually means rotations and translations of coordinates should transform predictions consistently.

## Role in Molecular Modeling

- Encode atoms, residues, and interactions as graphs.
- Respect 3D geometry in protein and ligand structures.
- Support coordinate prediction, pose refinement, and structure-aware ranking.

## Design Checks

- Which quantities are invariant and which are equivariant?
- Are edge features local, global, or both?
- Does the model preserve the symmetries needed by [[concepts/generative-models/flow-matching|flow matching]] or docking?

## Related

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[research/protein-modeling/mambafold|MambaFold]]
- [[concepts/sbdd/scoring-function|Scoring function]]
