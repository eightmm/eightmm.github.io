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

## Checks

- Is RMSD reported only after ligand identity and symmetry handling are correct?
- Are bond lengths, stereochemistry, ring planarity, and clashes checked?
- Are protein-ligand contacts plausible?
- Are pose quality, ranking, and affinity metrics reported separately?
- Are invalid generated poses excluded before using scores for downstream ranking claims?

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
