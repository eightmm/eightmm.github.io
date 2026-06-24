---
title: Protein-Ligand Complex
tags:
  - entities
  - protein-ligand
  - docking
---

# Protein-Ligand Complex

A protein-ligand complex combines a protein binding site, ligand conformation, and interaction geometry. It is a central object in structure-based AI.

## Questions

- Is the pose geometrically plausible?
- Does the [[concepts/sbdd/scoring-function|scoring function]] measure pose quality, binding affinity, or both?
- Which representation best preserves geometry and chemistry?

## Checks

- Is the binding site defined from a structure, pocket detector, or known ligand?
- Are clashes, bond geometry, chirality, and protein-ligand contacts validated?
- Does the model condition on the full protein, local pocket, surface, or residue graph?
- Are pose quality and binding affinity evaluated as separate targets?

## Related

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
