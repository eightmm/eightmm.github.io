---
title: Protein-Ligand Interaction
tags:
  - sbdd
  - protein-ligand
  - molecular-modeling
---

# Protein-Ligand Interaction

Protein-ligand interaction describes how a ligand and a protein binding site make contacts in 3D space. It connects docking, scoring, pose quality, binding affinity, and structure-based model interpretation.

A protein-ligand complex can be represented as:

$$
C = (P, L, X_P, X_L)
$$

where $P$ is the protein or pocket, $L$ is the ligand, and $X_P$, $X_L$ are their coordinates.

Pairwise protein-ligand distances are:

$$
d_{ij}
=
\lVert x_i^{(P)} - x_j^{(L)} \rVert_2
$$

for protein atom or residue coordinate $x_i^{(P)}$ and ligand atom coordinate $x_j^{(L)}$.

## Common Interaction Types

- Steric complementarity and shape fit.
- Hydrogen bonds.
- Hydrophobic contacts.
- Salt bridges and electrostatic interactions.
- Pi-stacking or pi-cation interactions.
- Metal coordination.
- Water-mediated contacts.

## Modeling Views

- Contact map between pocket atoms/residues and ligand atoms.
- Interaction fingerprint over predefined interaction rules.
- Geometric graph with protein and ligand nodes.
- Grid or point-cloud representation around the binding site.
- Learned pair representation for scoring or pose refinement.

## Checks

- Are hydrogens, protonation states, waters, metals, and cofactors handled consistently?
- Is the interaction evaluated from a predicted pose or an experimental structure?
- Are contacts physically plausible, or only close in Euclidean distance?
- Is the model using interaction evidence available at deployment time?
- Are pose quality and affinity claims separated?

## Related

- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
