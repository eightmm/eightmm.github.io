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

## Generation Contract

A conformer is tied to a protocol:

$$
X^{(k)}
\sim
G_{\pi}(M, s_k)
$$

where $G_{\pi}$ is the conformer generator, $\pi$ contains settings such as force field, embedding method, pruning, and minimization, and $s_k$ is a seed or stochastic state.

The contract should state:

- molecular identity used for generation;
- stereochemistry, tautomer, protonation, and charge state;
- number of conformers and pruning rule;
- energy minimization or ranking method;
- coordinate units and software version.

## Checks

- Was conformer generation seeded and versioned?
- Are protonation, tautomer, charge, and stereochemistry fixed before conformer generation?
- Is the model trained on crystal structures but evaluated on generated conformers?
- Are metrics stable across conformer ensembles?
- Does the task need ligand-only conformers or protein-bound poses?
- Is the conformer generation protocol part of the featurization contract?

## Evaluation Boundary

If the model consumes or predicts conformers, evaluation should separate:

- conformer validity: chemically plausible geometry;
- conformer coverage: whether low-energy or bioactive-like states are represented;
- pose correctness: placement relative to a protein pocket;
- property utility: whether conformers improve the downstream prediction.

## Conformer vs Pose

A ligand-only conformer is not the same as a bound pose:

$$
X_{\mathrm{conf}}
\ne
X_{\mathrm{pose}}(P)
$$

The pose depends on the protein or pocket context $P$. A model trained on bound poses but evaluated on ligand-only conformers is under a representation shift.

## Routing

| Question | Route |
| --- | --- |
| How is the molecule standardized before coordinates? | [Molecules](/bio/molecules) |
| Is this ligand-only geometry or protein-bound placement? | [Structure-based modeling](/bio/structure-based-ai) |
| Is a pose generated, scored, or filtered in a pocket? | [Docking](/bio/docking) |
| Does the model need equivariance or coordinate losses? | [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Is conformer generation used as featurization? | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract) |

## Failure Modes

- Train structures are experimental or bound poses, while inference uses generated conformers.
- Only the lowest-energy conformer is used even when the bioactive state differs.
- Conformer ensembles are generated with a different tautomer/protonation policy than training.
- Metrics ignore variance across conformers and report one lucky geometry.
- Coordinates are centered or aligned with deployment-unavailable context.

## Related

- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[bio/modeling-scope|Molecular Modeling Scope]]
- [[bio/molecules|Molecules]]
- [[bio/docking|Docking]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
- [[entities/ligand|Ligand]]
- [[entities/structure|Structure]]
