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

## Checks

- Is the target backbone, all-atom structure, contact map, distance map, or confidence score?
- Does training or evaluation use templates that may leak test structures?
- Are missing residues, alternate conformations, chain breaks, and ligands handled?
- Is structure cleaning and residue indexing consistent between train and evaluation?
- Is downstream use docking, design, function prediction, or representation transfer?

## Related

- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[research/protein-modeling/mambafold|MambaFold]]
