---
title: Geometry
aliases:
  - computational-biology/geometry
  - bio/geometry
tags:
  - computational-biology
  - geometry
unlisted: true
---


# Geometry

Geometry connects molecular modeling to graph, coordinate, symmetry, and equivariant modeling. Molecules and protein complexes are not only strings or graphs; many tasks depend on distances, angles, frames, and valid coordinate transformations.

$$
F(RX + t) = R F(X) + t
$$

This is the basic shape of equivariance for coordinate-valued outputs.

For invariant scalar outputs such as affinity or class probability, the usual requirement is:

$$
f(RX+t) = f(X)
$$

where $X$ is a coordinate set, $R$ is a rotation, and $t$ is a translation.

## Math and Geometry

- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/modalities/graph|Graph]]

## Geometric Deep Learning

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
- [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/tensor-field-network|Tensor field network]]

## Target Type Map

| Target | Symmetry Requirement | Examples |
| --- | --- | --- |
| Scalar | invariant | energy, affinity, class probability, ranking score |
| Vector / direction | equivariant | force, displacement, velocity, coordinate update |
| Coordinate set | equivariant up to permutation and rigid motion | ligand pose, atom coordinates, residue coordinates |
| Graph relation | often permutation equivariant/invariant | contact map, interaction edge, bond graph |

## Structure Representation Contract

Before writing a structure model note, separate the object, coordinate frame, and target:

| Field | Example Values | Why It Matters |
| --- | --- | --- |
| Unit | atom, residue, ligand, pocket, chain, complex | defines permutation and indexing rules |
| Coordinate source | experimental, predicted, docked, generated, relaxed | defines evidence strength and leakage risk |
| Frame | global frame, protein-centered, ligand-centered, local residue frame | controls invariance and equivariance requirements |
| Edges | bond, distance cutoff, kNN, residue contact, interaction edge | determines what information is visible |
| Target | scalar score, contact graph, coordinate update, pose, force | determines metric and symmetry |
| Alignment policy | none, receptor-aligned, ligand-aligned, symmetry-aware | determines how RMSD or error is computed |

For a protein-ligand complex:

$$
X
=
\left[
X_{\mathrm{protein}},
X_{\mathrm{ligand}}
\right]
$$

but the two parts often have different constraints: protein atoms may be fixed, side chains may move, and ligand torsions may change.

## Coordinate Features

| Feature | Formula | Use |
| --- | --- | --- |
| Distance | $d_{ij}=\lVert x_i-x_j\rVert_2$ | invariant edge feature |
| Direction | $u_{ij}=(x_j-x_i)/(d_{ij}+\epsilon)$ | equivariant message direction |
| Centered coordinate | $\tilde{x}_i=x_i-\frac{1}{N}\sum_j x_j$ | translation handling |
| Pairwise radial basis | $\psi(d_{ij})$ | smooth distance embedding |

Angles and torsions add information that distances alone may hide:

$$
\cos \theta_{ijk}
=
\frac{(x_i-x_j)^\top(x_k-x_j)}
{\lVert x_i-x_j\rVert_2\lVert x_k-x_j\rVert_2}
$$

A torsion angle depends on four ordered atoms and is sensitive to stereochemistry and conformer state. This is why molecule cleanup, residue indexing, and atom mapping belong in the geometry contract.

## Structure Tasks

- [[concepts/tasks/coordinate-prediction|Coordinate prediction]]
- [[concepts/tasks/graph-prediction|Graph prediction]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]

## Evaluation Boundary

| Claim | Metric Family | Required Context |
| --- | --- | --- |
| pose accuracy | RMSD, contact recovery, interaction recovery | atom mapping, receptor state, alignment policy |
| structure prediction | coordinate error, distance/contact accuracy | residue mapping, chain mapping, template policy |
| affinity or score | regression, ranking, enrichment | assay/source context and split unit |
| generation | validity, uniqueness, novelty, geometry checks | sampling budget and filtering policy |
| force or dynamics | vector error, energy consistency, trajectory stability | units, timestep, integrator, reference data |

Do not use a scalar benchmark score to imply geometric validity unless the geometry checks are reported.

## Checks

- Which outputs should be invariant and which should be equivariant?
- Are coordinates centered, aligned, or frame-dependent?
- Does the coordinate modeling contract match the claimed output and metric?
- Are edges constructed only from inputs available at inference time?
- Are chirality, stereochemistry, units, and atom/residue indexing preserved?
- Is the receptor, ligand, pocket, or full complex the actual modeled unit?
- Are failed generations, failed dockings, and filtered structures counted in the denominator?

## Related

- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[ai/architectures|Architectures]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
