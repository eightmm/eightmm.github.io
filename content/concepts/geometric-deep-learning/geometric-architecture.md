---
title: Geometric Architecture
tags:
  - geometric-deep-learning
  - architectures
---

# Geometric Architecture

A geometric architecture builds symmetry assumptions directly into a model. It is used when inputs include coordinates, directions, graphs, or physical structures.

The core design target is:

$$
F(g\cdot x) = \rho(g)F(x)
$$

where $g$ is a transformation such as rotation or translation, and $\rho(g)$ specifies how the output should transform.

## Design Choices

- Use invariant features such as distances for scalar outputs.
- Use equivariant coordinate or vector updates for structure prediction.
- Keep scalar, vector, and higher-order channels separate when needed.
- Use graph construction that respects the intended geometry.
- Match the final readout to the target: scalar score, vector field, coordinates, or distribution.

## Checks

- Is the target invariant or equivariant?
- Should reflections be included, or does chirality matter?
- Does preprocessing choose an arbitrary frame?
- Are generated coordinates evaluated for physical plausibility?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[research/structure-based-ai/index|Structure-based AI]]
