---
title: Pocket Representation
tags:
  - protein-modeling
  - sbdd
  - representation-learning
---

# Pocket Representation

Pocket representation describes how a protein binding site is converted into model input. The pocket can be represented as residues, atoms, surfaces, grids, graphs, coordinates, or learned embeddings.

A distance-defined pocket around ligand coordinates $X_L$ can be:

$$
\mathcal{P}_r
=
\{i \in P : \min_{j\in L}\lVert x_i-x_j\rVert_2 \le r\}
$$

where $r$ is a cutoff radius.

The pocket representation is then:

$$
z_{\mathcal{P}}
=
f_\theta(\mathcal{P}_r, X_{\mathcal{P}}, A_{\mathcal{P}})
$$

where $X_{\mathcal{P}}$ are coordinates or features and $A_{\mathcal{P}}$ is optional adjacency or contact structure.

## Common Representations

- Residue sequence window around binding residues.
- Atom-level graph with local coordinates.
- Residue-level graph or contact map.
- 3D grid or voxel representation.
- Surface patch with geometric and chemical features.
- Learned pocket embedding paired with ligand embedding.

## Key Risks

- Pocket extraction can leak ligand information if defined from the test ligand.
- Different cutoffs can change the apparent task.
- Missing side chains, cofactors, metals, and waters may alter interactions.
- Whole-protein context can matter even when the model only sees a local pocket.

## Checks

- Is the pocket known, predicted, transferred, or ligand-defined?
- What atoms, residues, chains, waters, metals, and cofactors are included?
- Does the representation preserve rotation/translation invariance or equivariance?
- Is the pocket representation available at deployment time?
- Is pocket similarity controlled across train and test?

## Related

- [[entities/pocket|Pocket]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
