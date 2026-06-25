---
title: Conformer
tags:
  - molecular-modeling
  - geometry
---

# Conformer

A conformer is a 3D arrangement of a molecule for a specific topology and stereochemical state. The same molecule can have multiple low-energy conformers.

Coordinates can be represented as:

$$
X = [x_1,\ldots,x_N]^\top,
\qquad
x_i\in\mathbb{R}^{3}
$$

where $N$ is the number of atoms.

An ensemble of conformers is:

$$
\mathcal{C}(M)=\{X^{(1)},\ldots,X^{(K)}\}
$$

## Checks

- Was conformer generation seeded and versioned?
- Are protonation, tautomer, charge, and stereochemistry fixed before conformer generation?
- Is the model trained on crystal structures but evaluated on generated conformers?
- Are metrics stable across conformer ensembles?
- Does the task need ligand-only conformers or protein-bound poses?
- Is the conformer generation protocol part of the featurization contract?

## Conformer vs Pose

A ligand-only conformer is not the same as a bound pose:

$$
X_{\mathrm{conf}}
\ne
X_{\mathrm{pose}}(P)
$$

The pose depends on the protein or pocket context $P$. A model trained on bound poses but evaluated on ligand-only conformers is under a representation shift.

## Related

- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
- [[entities/ligand|Ligand]]
- [[entities/structure|Structure]]
