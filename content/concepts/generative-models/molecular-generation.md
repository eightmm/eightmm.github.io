---
title: Molecular Generation
tags:
  - molecular-generation
  - generative-model
  - drug-discovery
---

# Molecular Generation

Molecular generation produces novel molecular structures—as strings, graphs, or 3D coordinates—often under constraints such as validity, synthesizability, or target properties.

Conditional molecular generation can be written as:

$$
m \sim p_\theta(m \mid c)
$$

where $m$ is a molecule and $c$ can represent a target property, scaffold, protein pocket, or design constraint.

## Why It Matters

- A core tool for de novo design in drug discovery and materials.
- Spans autoregressive, VAE, flow, and diffusion approaches.
- Useful output requires validity and property control, not just novelty.

## Checks

- Are generated molecules valid, novel, and synthesizable?
- Does the representation (SMILES, graph, 3D) suit the objective?
- Is property conditioning evaluated against held-out targets?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
