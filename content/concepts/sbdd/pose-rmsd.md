---
title: Pose RMSD
tags:
  - sbdd
  - docking
  - evaluation
---

# Pose RMSD

Pose RMSD measures how far a predicted ligand pose is from a reference pose after matching corresponding ligand atoms. It is a common docking metric, but it must be interpreted with symmetry, atom mapping, and receptor context in mind.

For predicted ligand coordinates $\hat{X}\in\mathbb{R}^{n\times 3}$ and reference coordinates $X\in\mathbb{R}^{n\times 3}$:

$$
\operatorname{RMSD}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\lVert \hat{x}_i - x_i\rVert_2^2
}
$$

The coordinates should be compared in the same receptor frame. For docking pose evaluation, the receptor is usually fixed and the ligand pose is compared directly in that coordinate system.

Use [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]] to record the frame, atom mapping, symmetry correction, loss, and metric before interpreting RMSD.

## Symmetry-Corrected RMSD

Ligands can have symmetric atoms. A naive atom order can make an equivalent pose look wrong. Symmetry-corrected RMSD minimizes over valid atom permutations:

$$
\operatorname{RMSD}_{\mathrm{sym}}
=
\min_{\pi\in\Pi}
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\lVert \hat{x}_{\pi(i)} - x_i\rVert_2^2
}
$$

where $\Pi$ is the set of chemically valid symmetric atom mappings.

## What It Measures

Pose RMSD mainly measures geometric agreement with a reference ligand pose. It does not directly measure:

- binding affinity
- interaction plausibility
- strain energy
- protonation correctness
- protein flexibility
- whether the pose is useful for virtual screening

This is why pose RMSD should be paired with [[concepts/sbdd/pose-quality|Pose quality]], [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]], and benchmark-specific checks.

## Common Threshold

A common heuristic is:

$$
\operatorname{RMSD} \le 2\text{ Å}
$$

for a successful docked pose. This is a convention, not a universal scientific law. The threshold can be misleading for very small ligands, symmetric ligands, flexible ligands, alternate binding modes, or noisy reference structures.

## Failure Modes

- Wrong atom mapping or ignored symmetry.
- Comparing poses after unintended ligand alignment instead of receptor-frame comparison.
- Treating a low RMSD as evidence of good affinity prediction.
- Ignoring alternative valid binding modes.
- Evaluating only the top-ranked pose when pose generation and scoring should be separated.
- Including leaked holo templates or close ligand analogs in training or template databases.

## Checks

- Is atom correspondence defined and symmetry-corrected?
- Are predicted and reference poses in the same receptor coordinate frame?
- Is the receptor structure comparable between prediction and reference?
- Is top-1 RMSD separated from best-of-N pose generation?
- Are pose RMSD, interaction quality, and affinity metrics reported separately?
- Is the benchmark protected against [[concepts/sbdd/template-leakage|Template leakage]]?
- Is the coordinate modeling contract explicit enough to reproduce the comparison?

## Related

- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[papers/sbdd/posebusters|PoseBusters]]
