---
title: Protein Structure Prediction
tags:
  - protein-modeling
  - structure-prediction
---

# Protein Structure Prediction

Protein structure prediction estimates 3D coordinates or structural constraints from sequence and optional context such as evolutionary information, templates, or learned representations.

A simplified sequence-to-structure view is:

$$
\hat{X}
= f_\theta(s_{1:L}, c)
$$

where $s_{1:L}$ is the residue sequence, $c$ is optional context, and $\hat{X}\in\mathbb{R}^{L\times 3}$ or atom-level coordinates are predicted.

Geometry losses often compare pairwise distances:

$$
\mathcal{L}_{\mathrm{dist}}
= \sum_{i,j}
\left(
\lVert \hat{x}_i-\hat{x}_j\rVert_2
- \lVert x_i-x_j\rVert_2
\right)^2
$$

Predictions can target different levels:

- contact map or distance map
- backbone coordinates
- all-atom coordinates
- confidence or error estimates
- complex or multimer structure
- pocket or binding-site geometry

The target choice changes evaluation. A good contact map does not imply all-atom pose quality, and a good backbone does not imply ligand-ready side-chain placement.

## Prediction Levels

| Target | Output | Evidence needed |
| --- | --- | --- |
| contact map | binary residue-pair contact | precision/recall by sequence separation |
| distance map | pairwise distances or bins | distance error and calibration |
| backbone | $N$, $C_\alpha$, $C$, $O$ coordinates | RMSD/TM-like structural agreement and local geometry |
| all-atom | backbone plus side chains | rotamer quality, clashes, bond geometry |
| confidence | per-residue or pairwise uncertainty | calibration against actual error |
| complex | multimer or protein-ligand/protein-protein arrangement | interface quality, chain mapping, partner leakage checks |

## Coordinate Contract

Coordinate predictions should state what transforms are irrelevant:

$$
\hat{X}' = R\hat{X}+t,
\qquad
X' = RX+t
$$

Global rotation and translation should not change the structural claim. If the model predicts vectors, forces, or coordinate updates, the output should transform equivariantly; if it predicts confidence or a scalar score, the output should be invariant.

## Downstream Boundary

| Downstream use | Extra requirement |
| --- | --- |
| docking | side-chain placement, pocket geometry, missing atoms, protonation states |
| binder or complex modeling | interface orientation, chain identity, template leakage control |
| function prediction | active-site residues, cofactors, conformational state |
| protein design | foldability and sequence-structure compatibility |
| representation learning | split-aware downstream evaluation, not only structure similarity |

## Checks

- Is the target backbone, all-atom structure, contact map, distance map, or confidence score?
- Does training or evaluation use templates that may leak test structures?
- Are missing residues, alternate conformations, chain breaks, and ligands handled?
- Is structure cleaning and residue indexing consistent between train and evaluation?
- Is downstream use docking, design, function prediction, or representation transfer?
- Is the confidence estimate calibrated to the reported error?
- Are monomer, multimer, and ligand-bound claims evaluated separately?

## Related

- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
