---
title: Molecular Dynamics
tags:
  - molecular-modeling
  - simulation
  - dynamics
---

# Molecular Dynamics

Molecular dynamics simulates time evolution of molecular coordinates under a physical model, usually by integrating equations of motion derived from an energy function.

A simplified form is:

$$
m_i \frac{d^2 x_i}{dt^2}
=
-\nabla_{x_i} E(X)
$$

where $x_i$ is atom $i$'s position, $m_i$ is mass, and $E(X)$ is the potential energy.

## What It Provides

Molecular dynamics is not just "more 3D data." It provides trajectories:

$$
\tau
=
(X_0, X_1,\ldots,X_T)
$$

These trajectories can support conformational analysis, stability checks, binding-mode hypotheses, flexible receptor context, or simulation-derived features. They also introduce strong protocol dependence.

## Protocol Contract

| Field | Meaning |
| --- | --- |
| System | ligand-only, protein-only, protein-ligand complex, membrane, solvent, ions |
| Force field | energy model and parameters |
| Ensemble | NVE, NVT, NPT, or other thermodynamic setup |
| Temperature/pressure | simulation condition |
| Time step and length | integration granularity and total trajectory duration |
| Initialization | starting structure, minimization, equilibration |
| Analysis window | which frames are used for metrics or features |

## ML Boundary

When molecular dynamics is used with AI, separate these roles:

| Role | Example | Check |
| --- | --- | --- |
| Data source | train on MD frames | split by molecule/protein, not by adjacent frames |
| Feature generator | average structure or dynamic descriptors | record trajectory protocol and frame selection |
| Label proxy | stability, residence, interaction frequency | label semantics may not match assay labels |
| Baseline | compare learned dynamics to simulation | integration cost and accuracy differ |
| Post hoc analysis | inspect a generated pose | do not turn qualitative inspection into a performance claim |

## Failure Modes

- Splitting adjacent frames across train and test, causing trajectory leakage.
- Treating short simulation stability as proof of binding affinity.
- Reporting dynamic features without the force-field and ensemble protocol.
- Mixing trajectories from different protocols without a dataset-card boundary.
- Ignoring that simulation time scale may not cover the biological event of interest.

## Related

- [[concepts/molecular-modeling/force-field|Force field]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
- [[math/dynamical-systems|Dynamical systems]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/evaluation/leakage|Leakage]]
