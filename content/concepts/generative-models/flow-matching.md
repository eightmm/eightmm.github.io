---
title: Flow Matching
tags:
  - generative-models
  - diffusion
  - flow-matching
---

# Flow Matching

Flow matching trains a model to predict a velocity field that transports samples from a simple distribution to a data distribution. It is closely related to diffusion and continuous normalizing flow perspectives.

The training objective is usually written as:

$$
\mathcal{L}
= \mathbb{E}_{t,x_t}
\left[
\lVert v_\theta(x_t,t) - u_t(x_t) \rVert^2
\right]
$$

Here $v_\theta$ is the learned velocity field and $u_t$ is the target velocity along a chosen probability path.

## Why It Matters

- Provides a generative modeling framework for continuous objects.
- Can be adapted to molecular coordinates and protein geometry.
- Gives a way to model trajectories instead of only final structures.

## Questions

- What path is used between noise and data?
- Which symmetries should the velocity field preserve?
- How does it interact with [[concepts/geometric-deep-learning/equivariant-gnn|equivariant GNN]] architectures?

## Related

- [[research/generative-models/index|Generative models]]
- [[research/protein-modeling/mambafold|MambaFold]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
