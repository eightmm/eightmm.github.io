---
title: Computational Biology Geometry Route
aliases:
  - computational-biology/geometry
  - bio/geometry
tags:
  - computational-biology
  - geometry
unlisted: true
---

# Computational Biology Geometry Route

For structure-modeling geometry, start from [[molecular-modeling/geometry-for-structure-modeling|Geometry for Structure Modeling]]. This page exists to route geometry questions into the right layer: pure math, geometric deep learning, or applied structure modeling.

Computational biology uses geometry when the object has coordinates, distance constraints, local frames, conformers, pockets, poses, or complexes. The same equation can mean different things depending on whether it is a math definition, a model architecture constraint, or a biological modeling claim.

$$
X \in \mathbb{R}^{n \times 3},
\qquad
D_{ij}=\lVert x_i-x_j\rVert_2,
\qquad
X' = RX+t
$$

Here $X$ can be atom coordinates, residue coordinates, ligand conformer coordinates, pocket points, or a protein-ligand complex. The route depends on what is being claimed about $X$.

Use the geometry notes by layer:

| Question | Go To |
| --- | --- |
| What pure math rule is involved? | [Geometry and Symmetry](/math/geometry-symmetry) |
| What does the structure-modeling workflow require? | [Geometry for Structure Modeling](/molecular-modeling/geometry-for-structure-modeling) |
| What is the coordinate/frame contract? | [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Is the claim pose-specific? | [Pose quality](/concepts/sbdd/pose-quality) |

## Routing Rules

| If the page is about | Keep it under |
| --- | --- |
| group action, metric space, distance, symmetry definition | [Math](/math) |
| invariance, equivariance, coordinate update, message passing | [Geometric Deep Learning](/concepts/geometric-deep-learning) |
| protein, ligand, pocket, complex, conformer, docking pose | [Computational Biology](/molecular-modeling) |
| pose plausibility, clash, strain, RMSD, interaction geometry | [Structure-Based Modeling](/molecular-modeling/structure-based) |

## Common Applied Questions

| Question | Better starting point |
| --- | --- |
| Does a generated ligand pose satisfy chemical geometry? | [Pose quality](/concepts/sbdd/pose-quality) |
| Is the model allowed to use the bound ligand to define the pocket? | [Pocket definition contract](/concepts/sbdd/pocket-definition-contract) |
| Should the output rotate when the input rotates? | [Equivariance](/concepts/geometric-deep-learning/equivariance) |
| Is RMSD the right metric for this claim? | [Pose quality](/concepts/sbdd/pose-quality) |
| Are distances enough, or are orientation and chirality required? | [Geometry for Structure Modeling](/molecular-modeling/geometry-for-structure-modeling) |

## Boundary

Keep protein, ligand, pocket, complex, pose, docking, and coordinate-source assumptions under Computational Biology. Keep reusable group, distance, invariance, and equivariance definitions under Math or Geometric Deep Learning. If a page depends on biological object identity or benchmark leakage risk, it belongs closer to Computational Biology than to pure Math.
