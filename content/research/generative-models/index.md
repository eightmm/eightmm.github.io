---
title: Generative Models
tags:
  - research
  - generative-models
---

# Generative Models

Generative model notes cover diffusion, flow matching, structure generation, molecular generation, and evaluation.

For this research area, the core question is conditional generation:

$$
x \sim p_\theta(x \mid c)
$$

where $x$ may be a molecule, protein sequence, structure, pose, or trajectory, and $c$ is the design condition.

## Topics

- [[concepts/generative-models/flow-matching|Flow matching]]

## Questions

- What object is generated: sequence, graph, coordinates, pose, or trajectory?
- Which constraints must be preserved?
- Which evaluation separates validity from usefulness?

## Related

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/protein-modeling/index|Protein modeling]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
