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

A path is usually defined by conditional distributions:

$$
x_t \sim p_t(x \mid x_0, x_1)
$$

where $x_0$ is sampled from a base distribution and $x_1$ from the data distribution. The model then defines an ODE sampler:

$$
\frac{dx_t}{dt}
=
v_\theta(x_t,t),
\qquad
x_0 \sim p_0
$$

The final sample is obtained by integrating from $t=0$ to $t=1$.

For a simple linear path:

$$
x_t = (1-t)x_0 + t x_1,
\qquad
u_t = x_1 - x_0
$$

but other paths can be chosen to fit geometry, noise schedules, or conditional constraints.

## Design Choices

- Probability path: linear, diffusion-like, optimal-transport inspired, or domain-specific.
- Vector field architecture: MLP, Transformer, GNN, or equivariant network.
- Conditioning: class, text, property, structure, pocket, or other context.
- Solver budget: number of function evaluations, step size, and error tolerance.

## Why It Matters

- Provides a generative modeling framework for continuous objects.
- Can be adapted to molecular coordinates and protein geometry.
- Gives a way to model trajectories instead of only final structures.

## Failure Modes

- A path that ignores constraints can generate invalid intermediate states.
- A non-equivariant velocity field can leak coordinate-frame assumptions.
- Solver cost can erase the benefit of a cleaner objective.
- Conditional generation can overfit the condition and reduce diversity.

## Checks

- What path is used between noise and data?
- Which symmetries should the velocity field preserve?
- How does it interact with [[concepts/geometric-deep-learning/equivariant-gnn|equivariant GNN]] architectures?
- Is the velocity target defined in a valid coordinate frame?
- Is sample quality reported together with solver budget?

## Related

- [[ai/generative-models|Generative models]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[research/protein-modeling/mambafold|MambaFold]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
