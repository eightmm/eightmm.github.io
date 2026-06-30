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

If the model learns the exact conditional velocity field, sampling follows:

$$
x_1
=
x_0
+
\int_0^1 v_\theta(x_t,t)\,dt
$$

where $x_0\sim p_0$ and the integrated endpoint should follow the data distribution.

## Objective Decomposition

For paper notes, record the path, sampling distribution, and target velocity:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{t\sim p(t),\,x_0\sim p_0,\,x_1\sim p_{\mathrm{data}},\,x_t\sim p_t(\cdot\mid x_0,x_1)}
\left[
\left\|v_\theta(x_t,t,c)-u_t(x_t\mid x_0,x_1,c)\right\|_2^2
\right].
$$

The notation hides important choices:

| Symbol | Meaning | Check |
| --- | --- | --- |
| $p_0$ | base distribution | noise, prior, scaffold, initial coordinates |
| $p_{\mathrm{data}}$ | target data distribution | train data, standardized objects, coordinate frame |
| $p(t)$ | time sampling | uniform, biased, schedule-weighted |
| $p_t$ | probability path | linear, OT-like, diffusion-like, domain-specific |
| $u_t$ | target velocity | analytic, simulated, estimated, conditional |
| $c$ | condition | class, text, property, pocket, protein, partial structure |

## Conditional Flow Matching

A common training view samples paired endpoints and trains on a conditional vector field:

$$
x_0\sim p_0,
\qquad
x_1\sim p_{\mathrm{data}},
\qquad
t\sim U(0,1)
$$

then:

$$
x_t \sim p_t(\cdot\mid x_0,x_1),
\qquad
u_t = u_t(x_t\mid x_0,x_1)
$$

The model sees $(x_t,t,c)$, not necessarily the endpoints. The learned field approximates the marginal velocity after averaging over possible endpoint pairs.

## Path Choice

The path is not a detail. It defines the target vector field and the difficulty of integration.

| Path | Benefit | Risk |
| --- | --- | --- |
| linear interpolation | simple analytic velocity | invalid intermediate objects |
| diffusion-like path | connects to score/diffusion tooling | schedule and parameterization matter |
| OT-inspired path | straighter transport and fewer steps | pairing or approximation assumptions |
| geometry-aware path | respects coordinates or constraints | harder target construction |
| conditional path | uses class, pocket, scaffold, or prompt | condition leakage or overfitting |

## Geometry Boundary

For coordinates, the velocity field must transform consistently:

$$
v_\theta(Rx+t,\tau,c')
=
R\,v_\theta(x,\tau,c)
$$

when rotations and translations should not change the physical meaning. If the model predicts scalar properties instead, the target is usually invariant rather than equivariant.

For molecular or protein coordinates, also state whether translation, rotation, permutation, chirality, bond constraints, and atom/residue identity are preserved along the path.

## Design Choices

- Probability path: linear, diffusion-like, optimal-transport inspired, or domain-specific.
- Vector field architecture: MLP, Transformer, GNN, or equivariant network.
- Conditioning: class, text, property, structure, pocket, or other context.
- Solver budget: number of function evaluations, step size, and error tolerance.
- Target parameterization: velocity, score, displacement, coordinate update, or domain-specific field.
- Symmetry rule: invariant, equivariant, permutation-aware, or frame-dependent.

## Why It Matters

- Provides a generative modeling framework for continuous objects.
- Can be adapted to molecular coordinates and protein geometry.
- Gives a way to model trajectories instead of only final structures.

## Failure Modes

- A path that ignores constraints can generate invalid intermediate states.
- A non-equivariant velocity field can leak coordinate-frame assumptions.
- Solver cost can erase the benefit of a cleaner objective.
- Conditional generation can overfit the condition and reduce diversity.
- Endpoint pairing can leak labels, templates, or target structures if not defined carefully.

## Checks

- What path is used between noise and data?
- Which symmetries should the velocity field preserve?
- How does it interact with [[concepts/geometric-deep-learning/equivariant-gnn|equivariant GNN]] architectures?
- Is the velocity target defined in a valid coordinate frame?
- Is sample quality reported together with solver budget?
- Are path choice, time sampling, and target velocity matched across baselines?
- Does the evaluation separate sample validity, condition satisfaction, diversity, and downstream utility?
- Are endpoint pairing, conditioning context, and target velocity public and reproducible?
- Does the path keep intermediate states inside a meaningful representation space?

## Related

- [[ai/generative-models|Generative models]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/coordinate-modeling-contract|Coordinate modeling contract]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
