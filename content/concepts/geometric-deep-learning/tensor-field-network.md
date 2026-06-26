---
title: Tensor Field Network
tags:
  - geometric-deep-learning
  - equivariance
  - tensor-field-network
---

# Tensor Field Network

A tensor field network is an equivariant neural architecture that represents features by their transformation type under 3D rotations.

An equivariant layer should satisfy:

$$
\Phi(D_{\mathrm{in}}(g)h)
= D_{\mathrm{out}}(g)\Phi(h)
$$

Here $D_{\mathrm{in}}$ and $D_{\mathrm{out}}$ describe how input and output feature types transform.

For a point set with coordinates $x_i\in\mathbb{R}^3$, a tensor field network updates features using messages that depend on relative geometry:

$$
h_i^{(\ell_{\mathrm{out}})}
\leftarrow
\sum_{j\in\mathcal{N}(i)}
\sum_{\ell_{\mathrm{in}},\ell_f}
W_{\ell_{\mathrm{in}},\ell_f\rightarrow\ell_{\mathrm{out}}}
\left(
h_j^{(\ell_{\mathrm{in}})}
\otimes
\phi_{\ell_f}(x_j-x_i)
\right)
$$

where $h_j^{(\ell_{\mathrm{in}})}$ is an input feature of type $\ell_{\mathrm{in}}$, $\phi_{\ell_f}$ is a geometric filter built from radial functions and [[concepts/geometric-deep-learning/spherical-harmonics|spherical harmonics]], and $W$ mixes only symmetry-compatible components.

## Feature Types

| Type | Example | Output Use |
| --- | --- | --- |
| $\ell=0$ scalar | atom type embedding, residue feature, distance-derived feature | energy, affinity, class probability, ranking |
| $\ell=1$ vector | direction, displacement, force-like feature | coordinate update, vector field, orientation-sensitive message |
| $\ell\ge2$ tensor | angular environment descriptor | high-order geometry when local orientation matters |

The model is usually more expensive than scalar message passing because tensor products, higher-order channels, and spherical harmonic filters increase memory and compute.

## Relation to Message Passing

Scalar graph message passing can be written as:

$$
h_i' = \sum_{j\in\mathcal{N}(i)} \psi(h_i,h_j,e_{ij})
$$

A tensor field network adds a representation-type contract:

$$
h_i^{(\ell)\prime}
\text{ must transform as }
D^\ell(R)h_i^{(\ell)}
$$

This makes it suitable for molecular structures, protein pockets, point clouds, and coordinate-aware generative models when the task depends on 3D orientation.

## Why It Matters

- It is an important design pattern for handling scalar, vector, and higher-order geometric features.
- The architecture connects neural message passing with [[concepts/geometric-deep-learning/spherical-harmonics|spherical harmonics]] and representation theory.
- It helps frame why equivariant models need structured feature channels instead of only scalar embeddings.
- It clarifies why some structure models are much heavier than ordinary [[concepts/architectures/gnn|GNNs]].

## When to Use

Use this family when:

- the input contains meaningful 3D coordinates;
- the output should be invariant or equivariant under rotations and translations;
- local angular geometry matters, not only pairwise distances;
- the extra compute is justified by the evaluation claim.

Avoid it as a default if the task only needs scalar graph features, a simple invariant readout, or a fast baseline.

## Checks

- Which tensor orders are used, and are they justified by the task?
- Are message passing operations equivariant at every layer?
- Is the added expressivity worth the compute and implementation complexity?
- Are final outputs converted to invariant or equivariant quantities correctly?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/so3|SO(3)]]
- [[concepts/geometric-deep-learning/irreducible-representation|Irreducible representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
