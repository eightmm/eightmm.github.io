---
title: Coordinate Modeling Contract
tags:
  - geometric-deep-learning
  - coordinates
  - structure-based-modeling
---

# Coordinate Modeling Contract

A coordinate modeling contract records how a model consumes, transforms, predicts, and evaluates 3D coordinates. It is especially important for structure-based modeling, docking, protein structure, molecular conformers, and geometric generative models.

The core question is:

$$
X
\xrightarrow{f_\theta}
\hat{Y}
\quad
\text{under}
\quad
X' = RX + \mathbf{1}t^\top
$$

where $X\in\mathbb{R}^{N\times 3}$ is an input coordinate set, $R$ is a rotation, and $t$ is a translation.

## Contract Fields

| Field | Question |
| --- | --- |
| Coordinate source | Are coordinates experimental, predicted, docked, generated, simulated, or cleaned? |
| Frame | Are coordinates in a global, centered, receptor, pocket, local residue, ligand, or learned frame? |
| Unit | Does one point represent atom, residue, bead, grid point, surface point, conformer, or pose candidate? |
| Ordering | Are atoms, residues, chains, or ligands indexed consistently across preprocessing and evaluation? |
| Transformation group | Should the model respect translation, rotation, reflection, permutation, or chirality constraints? |
| Input features | Which features are invariant scalars, equivariant vectors, or frame-dependent metadata? |
| Output type | Is the model predicting scalar, rank, contact, vector field, coordinate update, conformer, or pose? |
| Loss | Is the loss defined on coordinates, distances, frames, scores, velocities, or denoising targets? |
| Metric | Is evaluation RMSD, lDDT-style quality, clash/geometry check, interaction quality, enrichment, or affinity? |
| Leakage | Does preprocessing use known ligand pose, target-aligned frame, template, or post-hoc alignment unavailable at inference? |

## Target Symmetry

| Output | Expected Behavior | Example |
| --- | --- | --- |
| Scalar | invariant | affinity, energy, class probability, ranking score |
| Pairwise distance/contact | invariant to rigid motion, equivariant to permutation | distance map, contact map, interaction graph |
| Vector or field | equivariant | force, displacement, velocity, score field |
| Coordinate update | equivariant | atom update, residue update, ligand pose refinement |
| Pose or conformer | equivariant up to atom symmetry and rigid motion | docking pose, generated conformer |

For scalar prediction:

$$
f(RX+\mathbf{1}t^\top) = f(X)
$$

For coordinate prediction:

$$
F(RX+\mathbf{1}t^\top)
=
RF(X)+\mathbf{1}t^\top
$$

## Evaluation Alignment

Coordinate metrics need a frame and mapping rule.

| Metric | Must State |
| --- | --- |
| Pose RMSD | receptor frame, atom mapping, symmetry correction, top-1 vs best-of-N |
| Distance/contact quality | threshold, residue/atom unit, missing residues, chain policy |
| Geometry plausibility | bond lengths, angles, clashes, chirality, valence, strain |
| Interaction quality | pocket definition, interaction fingerprint, hydrogen/protonation policy |
| Affinity/ranking | assay context, split unit, baseline, calibration if probabilities are used |

## Structure-Based Modeling Risks

- A ligand-defined pocket frame can leak the answer when the ligand pose is unavailable at inference.
- Aligning predicted and reference ligands before RMSD can hide docking failure.
- A model can be equivariant internally but use non-equivariant preprocessing.
- Distance-only features can lose chirality or orientation needed for the task.
- Coordinate denoising loss can improve geometry while not improving ranking or affinity.
- Template-derived structures can make a test split easier than the claimed deployment setting.

## Checks

- What coordinate frame does the model receive?
- What transformation group should the input and output respect?
- Is the target invariant, equivariant, permutation-aware, or frame-dependent?
- Does the loss align with the reported coordinate metric?
- Is pose quality separated from scoring, affinity, and virtual screening utility?
- Are atom/residue mappings and symmetry corrections defined?
- Is preprocessing identical across train, validation, test, and inference where required?

## Related

- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[bio/geometry|Molecular modeling geometry]]
