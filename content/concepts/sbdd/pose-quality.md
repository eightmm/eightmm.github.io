---
title: Pose Quality
tags:
  - sbdd
  - docking
  - evaluation
---

# Pose Quality

Pose quality measures whether a predicted ligand pose is geometrically and chemically plausible in a protein binding site. It should be evaluated separately from binding affinity and interpreted alongside [[concepts/sbdd/pose-rmsd|Pose RMSD]].

A common native-pose comparison is ligand RMSD:

$$
\operatorname{RMSD}(X,\hat{X})
= \sqrt{\frac{1}{N}\sum_{i=1}^{N}
\lVert x_i-\hat{x}_i\rVert_2^2}
$$

where $X$ is the reference ligand pose, $\hat{X}$ is the predicted pose, and $N$ is the number of matched ligand atoms.

Validity can be treated as a conjunction of checks:

$$
\operatorname{valid}(\hat{X})
= \prod_{k=1}^{K}
\mathbf{1}[c_k(\hat{X})=1]
$$

## Quality Axes

Pose quality is multi-axis. A single RMSD number is not enough.

| Axis | Question | Typical Check |
| --- | --- | --- |
| atom mapping | are equivalent atoms and symmetries handled? | symmetry-corrected RMSD |
| internal geometry | is the ligand chemically plausible? | bond lengths, angles, chirality, ring planarity |
| protein-ligand geometry | does the pose avoid impossible contacts? | steric clashes, contact distances, interactions |
| pocket consistency | is the pose in the intended binding site? | pocket overlap and interaction fingerprint |
| receptor state | is the receptor compatible with the pose? | apo/holo state, side-chain clashes, missing residues |
| preparation state | are protonation, tautomer, charge, and hydrogens consistent? | chemical state contract |

## RMSD Boundary

RMSD needs an alignment and mapping policy:

$$
\operatorname{RMSD}_{\pi,R,t}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
Rx_i+t-\hat{x}_{\pi(i)}
\right\|_2^2
}
$$

where $\pi$ maps equivalent atoms and $(R,t)$ defines the alignment. For ligand poses in a fixed receptor frame, receptor alignment and ligand atom symmetry should be stated.

## Validity and Filtering

If invalid poses are filtered before scoring:

$$
\mathrm{score}_{\mathrm{kept}}
\neq
\mathrm{score}_{\mathrm{attempted}}
$$

Report both attempted and kept poses when the claim is about a generation or docking pipeline. Otherwise a strong score may reflect the filter rather than the model.

## Checks

- Is RMSD reported only after ligand identity and symmetry handling are correct?
- Are bond lengths, stereochemistry, ring planarity, and clashes checked?
- Are protein-ligand contacts plausible?
- Are pose quality, ranking, and affinity metrics reported separately?
- Are invalid generated poses excluded before using scores for downstream ranking claims?
- Are failed poses counted in the denominator?
- Is the receptor aligned, fixed, flexible, apo, holo, or predicted?

## Metric Boundary

Pose quality asks whether a candidate geometry is plausible:

$$
q = Q(P,L,\hat{X})
$$

It does not by itself prove binding affinity, selectivity, or screening utility. A native-like pose can still bind weakly, and a strong binder can be missed if the receptor or ligand state is wrong.

## Related

- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
