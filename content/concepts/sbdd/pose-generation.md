---
title: Pose Generation
tags:
  - sbdd
  - docking
  - generative-models
---

# Pose Generation

Pose generation produces candidate ligand geometries inside a protein pocket. It is the search or sampling part of docking, separate from scoring, affinity prediction, and virtual-screening ranking.

A pose generator can be written as a conditional distribution:

$$
\hat{X}
\sim
p_\theta(X \mid P, L, C)
$$

where $P$ is the protein or pocket, $L$ is the ligand, $C$ is optional context such as constraints or an initial geometry, and $\hat{X}$ is a generated ligand pose.

## What It Must Specify

- Binding-site definition: known pocket, predicted pocket, blind docking, or constraint-based search.
- Ligand state: standardized molecule, tautomer, protonation, stereochemistry, and conformer policy.
- Protein state: receptor structure source, chains, missing residues, side-chain handling, and flexibility assumption.
- Coordinate frame: how the pocket and ligand are centered, rotated, cropped, or augmented.
- Sampling policy: number of candidates, diversity, refinement, and filtering.

## Pose Variables

A ligand pose includes rigid and internal degrees of freedom:

$$
X = T(R, t, \tau; L)
$$

where $R$ is rotation, $t$ is translation, $\tau$ are torsion or conformer variables, and $T$ maps ligand identity and pose variables to 3D coordinates.

## Evaluation

Pose generation should be evaluated before scoring claims:

- Pose validity: bond lengths, chirality, clashes, ring geometry, and pocket overlap.
- Native-pose recovery when a valid reference pose exists.
- Diversity among generated poses.
- Robustness to receptor preparation and ligand state choices.
- Generalization under scaffold, protein-family, and template-leakage controls.

## Failure Modes

- Generating a chemically invalid ligand geometry.
- Placing the ligand in a leaked ligand-defined frame unavailable at deployment.
- Producing diverse-looking poses that all violate the same pocket constraint.
- Treating a high scoring pose as physically plausible without validation.
- Reporting affinity or enrichment without separating pose generation quality.

## Related

- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[papers/sbdd/posebusters|PoseBusters]]
