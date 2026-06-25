---
title: Molecular Graph
tags:
  - molecular-modeling
  - graphs
---

# Molecular Graph

A molecular graph represents atoms as nodes and bonds as edges. It is the natural input for graph neural networks and many molecular property models.

A molecule can be represented as:

$$
G = (V, E, X_V, X_E)
$$

where $V$ is the set of atoms, $E$ is the set of bonds, $X_V$ are atom features, and $X_E$ are bond features.

Message passing then updates atom states:

$$
h_i^{(t+1)}
= \phi_u
\left(
h_i^{(t)},
\sum_{j\in\mathcal{N}(i)}
\phi_m(h_i^{(t)}, h_j^{(t)}, e_{ij})
\right)
$$

## Checks

- Are hydrogens explicit or implicit?
- Are bond order, aromaticity, chirality, formal charge, and ring membership encoded?
- Does graph batching isolate molecules correctly?
- Does the graph lose conformer or stereochemical information needed by the task?
- Are graph features generated from the standardized molecule definition?

## Batching Boundary

For graph mini-batches, node tensors from many molecules are concatenated. A molecule-level readout must respect the batch index:

$$
h_G^{(b)}
=
\operatorname{pool}\{h_i: \operatorname{batch}(i)=b\}
$$

Pooling across all nodes without the batch boundary leaks information between examples.

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[entities/molecule|Molecule]]
