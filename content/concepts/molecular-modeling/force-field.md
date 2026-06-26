---
title: Force Field
tags:
  - molecular-modeling
  - physics
  - geometry
---

# Force Field

A force field is a parameterized energy function used to score molecular geometry. In molecular modeling, it provides a classical physics-based prior over bond lengths, angles, torsions, nonbonded contacts, and sometimes electrostatics.

A typical molecular mechanics energy has the form:

$$
E(X)
=
E_{\mathrm{bond}}
+ E_{\mathrm{angle}}
+ E_{\mathrm{torsion}}
+ E_{\mathrm{vdW}}
+ E_{\mathrm{elec}}
$$

where $X \in \mathbb{R}^{N \times 3}$ are atomic coordinates. The exact terms, parameters, atom types, charge model, and constraints depend on the chosen force field.

## Why It Matters

Force fields appear in conformer generation, docking preparation, molecular dynamics, energy minimization, pose refinement, and geometry sanity checks. A learned model may predict coordinates, but force-field-like checks are often still needed to diagnose impossible bond geometry, clashes, or strained conformations.

They should not be treated as ground truth. A force field is a model with assumptions and fitted parameters.

## Contract

Any public note using a force field should state:

| Field | Meaning |
| --- | --- |
| Force field family | which parameter set or model class is used |
| Atom typing | how atoms and bonds are assigned types |
| Charge policy | partial charge method, protonation state, and ion handling |
| Solvent/context | vacuum, implicit solvent, explicit solvent, or receptor context |
| Constraints | fixed atoms, rigid bonds, restraints, or cutoffs |
| Output use | minimization, ranking, filtering, MD, or feature generation |

The practical object is:

$$
(M, X, \pi_{\mathrm{ff}})
\rightarrow
E_{\pi_{\mathrm{ff}}}(X)
$$

where $M$ is the molecular identity and $\pi_{\mathrm{ff}}$ is the force-field protocol.

## Failure Modes

- Comparing energies from different force fields or charge policies as if they were calibrated.
- Minimizing a ligand state that does not match the assay or docking condition.
- Treating a low force-field energy as evidence of binding affinity.
- Reporting a learned pose without checking bond lengths, clashes, or stereochemistry.
- Using force-field minimization in evaluation without saying whether it changes model outputs.

## Checks

- Is the molecule standardized before atom typing?
- Are protonation, tautomer, charge, and stereochemistry fixed?
- Is the coordinate unit clear?
- Is energy used for filtering, ranking, or only geometry cleanup?
- Is the receptor or pocket included in the energy calculation?
- Are force-field artifacts separated from learned-model performance?

## Related

- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
