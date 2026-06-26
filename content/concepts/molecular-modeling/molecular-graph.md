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

## Atom and Bond Features

Common atom features include:

- atomic number or element;
- formal charge;
- aromaticity;
- hybridization;
- degree and valence;
- explicit or implicit hydrogens;
- chirality tag;
- ring membership.

Common bond features include:

- bond order;
- aromaticity;
- conjugation;
- ring membership;
- stereochemical direction.

The feature set is part of the model contract:

$$
\phi(M) = (X_V, X_E, A)
$$

where $A$ is the adjacency or edge index. If $\phi$ changes, cached features and splits may need regeneration.

## Batching Boundary

For graph mini-batches, node tensors from many molecules are concatenated. A molecule-level readout must respect the batch index:

$$
h_G^{(b)}
=
\operatorname{pool}\{h_i: \operatorname{batch}(i)=b\}
$$

Pooling across all nodes without the batch boundary leaks information between examples.

## Failure Modes

- Chiral tags or bond stereo are dropped while the label depends on stereochemistry.
- Formal charge and hydrogens are inconsistent with the protonation protocol.
- Global attention or normalization mixes nodes across molecules in a batch.
- A graph-only model is evaluated on a task requiring 3D conformer information.
- Features are generated from raw input while labels and splits use standardized identity.

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[entities/molecule|Molecule]]
