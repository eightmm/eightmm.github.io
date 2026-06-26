---
title: Geometric Deep Learning
tags:
  - geometric-deep-learning
  - geometry
---

# Geometric Deep Learning

Geometric deep learning is the AI layer built on top of [[concepts/math/geometry|geometry]] and [[concepts/math/symmetry-group|symmetry groups]]. It studies models that respect structure such as graphs, coordinates, transformations, and manifolds.

This section should not be a pure math textbook. Use it as the bridge from math foundations to neural architectures for proteins, molecules, 3D structures, vision, and graph-structured data.

The core question is how a model behaves under a transformation group $G$:

$$
f(g\cdot x) = \rho(g) f(x),
\qquad
g\in G
$$

Here $\rho(g)$ describes how the output representation should transform.

## Decision Pattern

For a geometric model, decide the contract before choosing the architecture:

$$
(\text{object}, \text{target}, \text{group}, \text{split})
\rightarrow
\text{feature and readout design}
$$

- Object: molecule, protein, protein-ligand complex, point cloud, graph, or structure.
- Target: scalar property, ranking, distance, vector field, coordinate update, pose, or generated structure.
- Group: permutation, translation, rotation, reflection, SO(3), SE(3), or E(3).
- Split: ligand scaffold, protein family, complex pair, assay/source, or time when relevant.

The group choice should match both the data and the deployment setting. A symmetry enforced by preprocessing is only valid if the same information is available at inference time.

## Math Background

| Need | Start |
| --- | --- |
| distance, angle, coordinate basics | [Geometry](/concepts/math/geometry) |
| group actions and symmetry | [Symmetry group](/concepts/math/symmetry-group) |
| vectors, matrices, bases, eigenspaces | [Linear algebra](/concepts/math/linear-algebra) |

## Symmetry

| Question | Start |
| --- | --- |
| should the output stay the same after a transform? | [Invariance](/concepts/geometric-deep-learning/invariance), [Invariant feature](/concepts/geometric-deep-learning/invariant-feature) |
| should the output transform with the input? | [Equivariance](/concepts/geometric-deep-learning/equivariance), [Equivariant feature](/concepts/geometric-deep-learning/equivariant-feature) |
| rotations only | [SO(3)](/concepts/geometric-deep-learning/so3) |
| rotations and translations | [SE(3)](/concepts/geometric-deep-learning/se3) |
| rotations, translations, and reflections | [E(3)](/concepts/geometric-deep-learning/e3) |

## Coordinates and Distances

| Need | Start | Typical Output |
| --- | --- | --- |
| declare coordinate source and frame | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame) | valid input contract |
| use distances without orientation | [Distance geometry](/concepts/geometric-deep-learning/distance-geometry) | invariant scalar features |
| predict or refine coordinates | [Coordinate update](/concepts/geometric-deep-learning/coordinate-update) | equivariant vector/coordinate output |

## Representations

| Need | Start | Use When |
| --- | --- | --- |
| angular basis on directions | [Spherical harmonics](/concepts/geometric-deep-learning/spherical-harmonics) | local orientation or angular structure matters |
| scalar/vector/tensor feature types | [Irreducible representation](/concepts/geometric-deep-learning/irreducible-representation) | feature channels must transform predictably |
| equivariant message passing with typed channels | [Tensor Field Network](/concepts/geometric-deep-learning/tensor-field-network) | higher-order geometric expressivity is worth the cost |

## Models and Operations

| Need | Start |
| --- | --- |
| choose a geometric model family | [Geometric architecture](/concepts/geometric-deep-learning/geometric-architecture) |
| graph neural network with symmetry constraints | [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn) |

## Public-Safe Checks

- State the coordinate source: experimental, predicted, docked, generated, simulated, or conformer-generated.
- State whether chirality and stereochemistry are retained.
- State whether graph construction uses only deployment-available inputs.
- State whether scalar outputs are invariant and coordinate/vector outputs are equivariant.
- Avoid private structures, unpublished results, host paths, and internal benchmark names.

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
