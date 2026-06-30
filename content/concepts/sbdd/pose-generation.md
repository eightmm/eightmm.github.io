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

## Sampling Contract

Pose generation is a conditional sampling problem:

$$
\{\hat{X}^{(k)}\}_{k=1}^{K}
\sim
\pi_\theta(\cdot\mid P,L,C)
$$

where $K$ is the number of attempted poses. Evaluation should preserve the denominator:

$$
\text{valid-pose rate}
=
\frac{\#\text{valid generated poses}}
{\#\text{attempted poses}}
$$

Reporting only the best retained pose hides invalid generation, failed refinement, and filtering cost.

## Generator Families

| Family | Pose Search Shape | Risk |
| --- | --- | --- |
| classical docking search | heuristic/global-local search over pose variables | scoring and search failures are entangled |
| conformer + placement | generate conformers then align/place | conformer library bias |
| diffusion or flow | iterative coordinate/torsion denoising or transport | sampler budget and geometry validity |
| autoregressive torsion | sequential torsion or atom placement | order dependence and accumulated errors |
| refinement model | starts from initial pose and updates coordinates | depends on initial pose distribution |

The generator family should be separated from the scorer. A better score can rescue poor generation, and a better generator can be hidden by a weak score.

## Coordinate Frame Boundary

If the pocket is centered or aligned using the reference ligand, the task is no longer blind docking. A public note should state what frame is available at inference:

| Frame Source | Deployment Meaning |
| --- | --- |
| receptor-only pocket center | usable for known pocket docking |
| predicted pocket center | depends on pocket predictor quality |
| ligand-defined box | can leak reference pose unless available from protocol |
| template ligand alignment | template-leakage risk |
| random or canonical frame | requires equivariant or augmentation-aware model |

For coordinate-generative models, the output should transform consistently under rigid motion:

$$
\hat{X}(RP+t,L)
=
R\hat{X}(P,L)+t
$$

when the problem is rotation/translation equivariant.

## Evaluation

Pose generation should be evaluated before scoring claims:

- Pose validity: bond lengths, chirality, clashes, ring geometry, and pocket overlap.
- Native-pose recovery when a valid reference pose exists.
- Diversity among generated poses.
- Robustness to receptor preparation and ligand state choices.
- Generalization under scaffold, protein-family, and template-leakage controls.

## Pose Selection

If multiple poses are generated, the reported pose usually depends on a selection rule:

$$
k^\*
=
\arg\max_k S(P,L,\hat{X}^{(k)})
$$

where $S$ can be a scoring function, confidence model, energy, or heuristic filter. Pose generation quality and pose selection quality should be evaluated separately when possible.

## Failure Modes

- Generating a chemically invalid ligand geometry.
- Placing the ligand in a leaked ligand-defined frame unavailable at deployment.
- Producing diverse-looking poses that all violate the same pocket constraint.
- Treating a high scoring pose as physically plausible without validation.
- Reporting affinity or enrichment without separating pose generation quality.
- Counting only selected poses rather than all attempted poses.
- Using a pocket box, template, or initial pose unavailable in deployment.

## Checks

- How many candidate poses are attempted, retained, refined, and scored?
- Is the coordinate frame receptor-only, predicted-pocket, template-based, or ligand-defined?
- Are ligand chemical state and receptor preparation fixed before generation?
- Is the generator evaluated separately from the selector/scorer?
- Are invalid, filtered, and failed samples included in the denominator?
- Are symmetry-aware atom mapping and pose validity checks used?

## Related

- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/pocket-definition-contract|Pocket definition contract]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[papers/sbdd/posebusters|PoseBusters]]
