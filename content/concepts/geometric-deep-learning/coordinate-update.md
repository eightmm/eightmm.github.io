---
title: Coordinate Update
tags:
  - geometric-deep-learning
  - coordinates
  - equivariance
---

# Coordinate Update

A coordinate update is a model step that changes point positions while preserving the intended geometric symmetry.

A translation-invariant coordinate update often uses relative vectors:

$$
x_i' = x_i
+ \sum_{j\in \mathcal{N}(i)}
\alpha_{ij}(x_j - x_i)
$$

If $\alpha_{ij}$ is invariant to global rotations and translations, the coordinate update can be equivariant.

One common message-passing form is:

$$
m_{ij}
=
\phi_m(h_i,h_j,d_{ij},e_{ij}),
\qquad
d_{ij}=\lVert x_i-x_j\rVert_2
$$

$$
\Delta x_i
=
\sum_{j\in\mathcal{N}(i)}
\phi_x(m_{ij})(x_i-x_j),
\qquad
x_i^{\mathrm{new}} = x_i + \Delta x_i
$$

If $\phi_x(m_{ij})$ is scalar and $m_{ij}$ is invariant, then $\Delta x_i$ rotates as a vector:

$$
\Delta x_i' = R\Delta x_i
$$

and the update is rotation equivariant.

## Stability

Coordinate updates are often repeated across layers, refinement steps, or sampling steps. Stability depends on step scale, neighborhood definition, normalization, and whether updates preserve physically meaningful constraints.

One way to make the step size explicit is:

$$
x_i^{(l+1)}
=
x_i^{(l)}
+ \eta_l \Delta x_i^{(l)}
$$

where $\eta_l$ may be fixed, learned, scheduled, or controlled by a numerical solver. Large $\eta_l$ can improve speed but create clashes, drift, or invalid geometry.

## Update Types

| Update | Form | Use |
| --- | --- | --- |
| displacement | $x_i' = x_i + \Delta x_i$ | refinement, denoising, coordinate prediction |
| velocity field | $\frac{dx}{dt}=v_\theta(x,t)$ | flow matching, probability-flow ODEs |
| score field | $\nabla_x \log p_t(x)$ | score-based diffusion and denoising |
| force-like field | $F_i=-\nabla_{x_i}E$ | energy-based or physics-inspired models |
| frame update | rotate/translate local frame | residue frames, rigid body refinement |

## Constraint Boundary

Coordinate updates can optimize a loss while violating constraints:

$$
\mathcal{L}_{\mathrm{coord}}
\downarrow
\quad\not\Rightarrow\quad
\text{valid chemistry or structure}
$$

For molecules and proteins, record which constraints are enforced or only evaluated: bond lengths, angles, chirality, steric clashes, chain connectivity, pocket constraints, atom mapping, and symmetry.

## Why It Matters

- Coordinate updates appear in structure refinement, docking, diffusion, and flow-based generation.
- Updates should be designed so translated or rotated inputs produce translated or rotated outputs.
- Separating scalar features from coordinate movement helps audit equivariance assumptions.

## Failure Modes

- Absolute coordinates enter the scalar message, breaking translation behavior.
- A vector update is normalized or clipped in a way that depends on a global frame.
- Large repeated updates create invalid geometries, atom clashes, or drift.
- Coordinate targets are rotated during augmentation but vector/force labels are not rotated consistently.

## Checks

- Are coordinate deltas built from relative vectors or arbitrary absolute positions?
- Does the update preserve translation and rotation behavior?
- Are scalar features allowed to influence movement without breaking symmetry?
- Is the update stable under repeated refinement or sampling steps?
- Are units consistent, such as Angstrom versus nanometer?
- Are generated structures evaluated for validity, not only coordinate loss?
- Is the step size, solver, or number of refinement steps fixed across comparisons?
- Are chemistry, chain, or rigid-body constraints enforced during the update?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
